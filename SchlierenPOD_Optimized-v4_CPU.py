# -*- coding: utf-8 -*-
"""
Created on Sun Jul 9 14:25:16 2023

@author: Hady-PC
"""
# -*- coding: utf-8 -*-
"""
Optimized Snapshot POD for very large image sets (~11k frames @ 900x960)
- Two-pass streaming over images (no giant image matrix in RAM)
- Blocked accumulation of temporal covariance R = X_f X_f^T
- Chunked computation / optional writing of spatial modes (Phi) to HDF5
- Reconstructs and saves a limited number of snapshots
- Produces the same outputs (plots + files) as the original, but scales


Notes:
- Expect the eigen-decomposition of an 11k x 11k matrix to take time. This code
  is written for robustness and memory efficiency; speed comes from BLAS and streaming.
- The optional full Phi (spatial modes) HDF5 will be ~ (n_snapshots * H*W * 4) bytes
  for float32; with 11k frames and 864k pixels that's ~38 GB. Keep write_full_phi=True
  only if you actually need the full matrix on disk. Otherwise, we compute and save
  only the selected modes' images and do reconstructions without materializing Phi fully.

"""

import os
import sys
import glob
import time
# import math
import h5py
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ------------------------
# Configuration
# ------------------------

# How many images to use (-1 = all in folder)
images_to_read = -1

# How many spatial modes images to save explicitly
num_modes_to_save = 20
start_mode = 1  # inclusive start index for reconstructions

# Path to images (PNG or other extensions ok if you change glob)
imgs_path = r"raw_images\*.png"

# Directory that contains Avg/00.png (mask image)
# (Automatically inferred from imgs_path; expects Avg/00.png next to the "clean" dir)

# Memory/Performance knobs
block_size = 50            # number of snapshots per block when streaming; 64–256 is typical
use_float32_R = True         # accumulate R in float32 (fast, lighter); switch to False for float64
write_full_phi = False       # WARNING: True -> writes ~38 GB for 11k frames (float32)
phi_chunk_cols = 50        # how many modes (columns) to compute at once when writing full Phi
reconst_acum_ratio = 100
min_snr = 0.6

# Plot style (matches your original)
plt.rcParams.update({'font.size': 30})
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"

# ------------------------
# Utilities
# ------------------------

def list_image_files(path_pattern, limit=-1):
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {path_pattern}")
    if limit == -1 or limit > len(files):
        limit = len(files)
    return files[:limit]


def read_gray_float32(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    return img.astype(np.float32)


def pass1_compute_mean(files):
    """Compute mean image in a single streaming pass."""
    first = read_gray_float32(files[0])
    H, W = first.shape
    acc = np.zeros_like(first, dtype=np.float64)
    for i, fp in enumerate(files):
        img = read_gray_float32(fp)
        if img.shape != (H, W):
            raise ValueError(f"Image size mismatch at {fp}: got {img.shape}, expected {(H, W)}")
        acc += img
        if (i+1) % 1000 == 0 or i == len(files)-1:
            sys.stdout.write(f"\rMean pass: {i+1}/{len(files)}")
            sys.stdout.flush()
    mean_img = (acc / len(files)).astype(np.float32)
    print("\nMean computed.")
    return mean_img, (H, W)

def build_R_double_block(files, mean_img, block_size=64, use_float32=True):
    n = len(files)
    dtype = np.float32 if use_float32 else np.float64
    R = np.zeros((n, n), dtype=dtype)
    H, W = mean_img.shape
    P = H * W
    mean_flat = mean_img.reshape(-1)

    # Pre-read nothing; stream on the fly to keep memory modest
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        b1 = i1 - i0
        # load block A
        A = np.empty((b1, P), dtype=np.float32)
        for ii, idx in enumerate(range(i0, i1)):
            A[ii, :] = read_gray_float32(files[idx]).reshape(-1) - mean_flat
        # inner loop for j >= i0 to exploit symmetry
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            b2 = j1 - j0
            # load block B
            B = np.empty((b2, P), dtype=np.float32)
            for jj, jdx in enumerate(range(j0, j1)):
                B[jj, :] = read_gray_float32(files[jdx]).reshape(-1) - mean_flat
            # accumulate
            # A (b1,P) @ B^T (P,b2) => (b1,b2)
            R_block = A @ B.T
            R[i0:i1, j0:j1] += R_block.astype(dtype, copy=False)
            if j0 != i0:
                # mirror to lower triangle
                R[j0:j1, i0:i1] += R_block.T.astype(dtype, copy=False)
        sys.stdout.write(f"\rR blocks up to rows {i1}/{n}")
        sys.stdout.flush()
    print("\nR built.")
    return R


def eigh_sorted(R):
    # numpy.linalg.eigh returns ascending; we reverse
    w, V = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    return w.astype(np.float32), V.astype(np.float32)


def save_eig_outputs(evals, evecs, out_dir, reconst_acum_ratio=95):
    os.makedirs(out_dir, exist_ok=True)
    # energy metrics
    sigma = np.sum(evals)
    acum = np.cumsum(evals)
    acum_ratio = (acum / sigma) * 100.0
    energy_ratio = (evals / sigma) * 100.0
    # 95% threshold
    n_modes = int(np.searchsorted(acum_ratio, reconst_acum_ratio))

    # Save CSV for eigenvalues & ratios
    header = "mode_index,eigenvalue,energy_ratio_percent,accum_ratio_percent"
    modes_idx = np.arange(len(evals), dtype=np.int32)
    table = np.column_stack([modes_idx, evals, energy_ratio, acum_ratio])
    np.savetxt(os.path.join(out_dir, 'eigenvalues.csv'), table, delimiter=",",
               header=header, comments='')

    # Save eigenvectors (V) in HDF5 (safer than huge CSV)
    with h5py.File(os.path.join(out_dir, 'eigenvectors.h5'), 'w') as h5f:
        h5f.create_dataset('V', data=evecs, compression='gzip', compression_opts=4)

    # Plot
    fig, ax1 = plt.subplots(figsize=(20, 14))
    ax2 = ax1.twinx()
    x = np.arange(1, len(evals) + 1)
    ax1.semilogx(x, energy_ratio, label=r'eigenvalues ratio', lw=3)
    ax2.semilogx(x, acum_ratio, label=r'Accumulative sum ratio',
                 lw=3, color='tab:orange')
    ax1.set_ylabel(r"$\lambda_i/\Sigma\lambda_i$ (\%)")
    ax2.set_ylabel(r"$\Sigma_i\lambda_i/\Sigma\lambda_i$ (\%)")
    ax1.set_xlabel("Modes")
    ax1.grid(True, which='major', color='#A0A0A0', linestyle='-')
    ax1.minorticks_on()
    ax1.grid(True, which='minor', color='#E0E0E0', linestyle='-', alpha=0.2)
    fig.savefig(os.path.join(out_dir, 
                             'Normalized eigenvalues in descending order for each POD mode.jpg'),
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Number of modes for {reconst_acum_ratio}% reconstruction: {n_modes}")
    return energy_ratio, acum_ratio, n_modes

def compute_phi_selected(files, mean_img, V, evals, mode_indices, block_size=64):
    """
    Compute Phi[:, mode_indices] without materializing full X_f.
    Returns Phi_sel (P, m) where m = len(mode_indices).
    """
    H, W = mean_img.shape
    P = H * W
    mean_flat = mean_img.reshape(-1)
    m = len(mode_indices)
    Phi_sel = np.zeros((P, m), dtype=np.float32)
    inv_sqrt = (1.0 / np.sqrt(np.maximum(evals[mode_indices], 1e-30))).astype(np.float32)

    n = len(files)
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        b = i1 - i0
        # load block B (b, P)
        B = np.empty((b, P), dtype=np.float32)
        for ii, idx in enumerate(range(i0, i1)):
            B[ii, :] = read_gray_float32(files[idx]).reshape(-1) - mean_flat
        # V_block (b, m)
        V_block = V[i0:i1, mode_indices].astype(np.float32, copy=False)
        # accumulate: B^T @ V_block -> (P, m)
        Phi_sel += B.T @ V_block
        sys.stdout.write(f"\rPhi selected accumulation: rows {i1}/{n}")
        sys.stdout.flush()
    print("\nPhi selected accumulated.")
    # scale by Lambda^{-1/2}
    Phi_sel *= inv_sqrt[np.newaxis, :]
    return Phi_sel.reshape(H*W, m)

def compute_mode_snr(Phi, H, W, mask=None):
    """
    Compute SNR for each spatial mode.

    Phi: (pixels, m) POD spatial modes.
    H, W: image dimensions.
    mask: optional binary mask (H, W). If given, only compute inside flow region.
    """
    m = Phi.shape[1]
    snr_vals = []
    for i in range(m):
        mode_img = Phi[:, i].reshape(H, W)
        if mask is not None:
            vals = mode_img[mask > 0]
        else:
            vals = mode_img.ravel()
        mu = np.mean(np.abs(vals))
        sigma = np.std(vals) + 1e-8
        snr_vals.append(mu / sigma)
    return np.array(snr_vals)

def select_modes(indices, snr_vals, energy_threshold=0.9, snr_min=2.0):
    """
    Select modes until reaching given cumulative energy,
    but only keep those with SNR above snr_min.
    """
    # total_energy = np.sum(lambdas)
    # cum_energy = 0.0
    selected = []
    for i, indx in enumerate(indices):
        # cum_energy += lam
        # if (cum_energy / total_energy)*100 < energy_threshold:
        #     break
        if snr_vals[i] <= snr_min:
            selected.append(indx)
    return selected

def write_phi(files, mean_img, V, evals, out_path, block_size=64, chunk_cols=64):
    """
    Optionally compute and write the FULL Phi (P, n) to HDF5 without
    holding it all in RAM. Writes dataset 'data' shaped (n, P) to match
    original (Y_snap = Phi^T).
    WARNING: This can be ~38 GB for 11k frames at 900x960.
    """
    n = V.shape[0]
    H, W = mean_img.shape
    P = H * W
    mean_flat = mean_img.reshape(-1)

    with h5py.File(out_path, 'w') as h5f:
        dset = h5f.create_dataset('data', shape=(n, P), dtype='float32',
                                  chunks=(max(1, 65536//P), P))
        # process columns of V in chunks
        for c0 in range(0, n, chunk_cols):
            c1 = min(c0 + chunk_cols, n)
            cols = np.arange(c0, c1)
            inv_sqrt = (1.0 / np.sqrt(np.maximum(evals[cols], 1e-30))).astype(np.float32)
            # accumulate Phi_chunk (P, m)
            m = len(cols)
            Phi_chunk = np.zeros((P, m), dtype=np.float32)
            for i0 in range(0, n, block_size):
                i1 = min(i0 + block_size, n)
                b = i1 - i0
                B = np.empty((b, P), dtype=np.float32)
                for ii, idx in enumerate(range(i0, i1)):
                    B[ii, :] = read_gray_float32(files[idx]).reshape(-1) - mean_flat
                V_block = V[i0:i1, cols].astype(np.float32, copy=False)
                Phi_chunk += B.T @ V_block
            # scale columns by inv sqrt lambdas
            Phi_chunk *= inv_sqrt[np.newaxis, :]
            # write Y_snap rows for these modes (Phi^T)
            dset[c0:c1, :] = Phi_chunk.T
            sys.stdout.write(f"\rFull Phi written cols {c1}/{n}")
            sys.stdout.flush()
    print("\nFull Phi written to HDF5.")


def save_mode_images(Phi_sel, mode_indices, H, W, avg_img, maskimg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for j, k in enumerate(mode_indices):
        fig, ax = plt.subplots(figsize=(20, 14))
        spimg = ax.imshow(Phi_sel[:, j].reshape(H, W), cmap="jet")
        if maskimg is not None:
            ax.imshow(maskimg, extent=[0, avg_img.shape[1],
                                       avg_img.shape[0], 0])
        ax.set_title(f'Special Mode No. {k}')
        fig.colorbar(spimg, ax=ax, format='%.0e')
        fig.savefig(os.path.join(out_dir, f"{k:04d}.png"), bbox_inches='tight',
                    pad_inches=0.1)
        plt.close(fig)


def reconstruct_and_save(Phi_sel, V, mode_indices, avg_img, maskimg, out_dir,
                         lambdas, start_idx=0, limit=100):
    """
    Reconstruct a limited number of snapshots using only the selected modes.
    Includes energy-based normalization so avg_img (≈80% energy) does not dominate.
    
    Parameters
    ----------
    Phi_sel (ndarray): POD spatial modes (flattened basis vectors).
    V (ndarray)      : Time coefficients (snapshots x modes).
    mode_indices (list or ndarray): Selected modes for reconstruction.
    avg_img (ndarray): Mean image (H, W).
    maskimg (ndarray or None): Optional mask overlay.
    out_dir (str)    : Output directory for saving reconstructed frames.
    lambdas (ndarray): POD eigenvalues (energies).
    start_idx (int)  : Starting snapshot index.
    limit (int)      : Number of snapshots to reconstruct (-1 means all).
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = avg_img.shape
    avg_float = avg_img.astype(np.float32)
    n = V.shape[0]

    total_energy = np.sum(lambdas)
    energy_avg = lambdas[0] / total_energy
    energy_fluct = np.sum(lambdas[mode_indices]) / total_energy

    end_idx = n if limit < 0 else min(n, start_idx + limit)
    for j in range(start_idx, end_idx):
        coeffs = V[j, mode_indices].astype(np.float32)
        rec_flat = Phi_sel @ coeffs
        rec = rec_flat.reshape(H, W)

        # Normalize separately
        avg_norm = (avg_float - np.min(avg_float)) / (np.max(avg_float) - np.min(avg_float) + 1e-8)
        rec_norm = (rec - np.min(rec)) / (np.max(rec) - np.min(rec) + 1e-8)

        # Weighted combination
        combined = 255 * (energy_avg * avg_norm + energy_fluct * rec_norm)
        # combined = 255 * (energy_fluct * rec_norm)
        combined = combined.astype(np.uint8)

        # Plot and save
        fig, ax = plt.subplots(figsize=(20, 14))
        spimg = ax.imshow(combined, cmap="gray", vmin=0, vmax=255)
        if maskimg is not None:
            ax.imshow(maskimg, extent=[0, W, H, 0])
        ax.set_title(f'Reconstructed Snapshot No. {j}')
        fig.colorbar(spimg, ax=ax, format='%d')
        fig.savefig(os.path.join(out_dir, f"R-{j:04d}.png"),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        if (j - start_idx + 1) % 5 == 0:
            sys.stdout.write(f"\rSaved reconstructions: {j - start_idx + 1}/{end_idx - start_idx}")
            sys.stdout.flush()

    print("\nReconstruction images saved.")

# ------------------------
# Main
# ------------------------
if __name__ == '__main__':
    t0 = time.time()

    # Derive directories
    folders = imgs_path.split("\\")
    file_directory = ''
    for i in range(len(folders)-1):
        file_directory += (folders[i] + '\\')
    pod_analysis_dir = os.path.join(file_directory, "PODAnalysis")
    pod_reconstruction_dir = os.path.join(file_directory, "PODReconstruction")
    os.makedirs(pod_analysis_dir, exist_ok=True)
    os.makedirs(pod_reconstruction_dir, exist_ok=True)

    # Mask image (optional)
    mask_path = os.path.join(file_directory, 'Avg', '00.png')
    maskimg = plt.imread(mask_path) if os.path.exists(mask_path) else None

    # File list
    files = list_image_files(imgs_path, images_to_read)
    n_snapshots = len(files)
    print(f"Found {n_snapshots} images.")

    # Pass 1: mean image
    avg_img, (H, W) = pass1_compute_mean(files)

    # Pass 2: build R (double-block streaming)
    print('Building temporal covariance R via streaming blocks...')
    R = build_R_double_block(files, avg_img, block_size=block_size,
                             use_float32=use_float32_R)

    # Eigendecomposition
    print('Eigendecomposition of R (on CPU)...')
    evals, V = eigh_sorted(R)
    del R  # free

    # Save eigen results + energy plot
    energy_ratio, acum_ratio, n_modes = save_eig_outputs(evals, V,
                                                         pod_analysis_dir,
                                                         reconst_acum_ratio)

    # Determine which modes to visualize (same pattern as original)
    mode_indices = list(range(num_modes_to_save))
    # add every 100th mode after num_modes_to_save up to n_snapshots
    mode_indices += list(range(num_modes_to_save, 100, 10))
    mode_indices += list(range(100, n_snapshots, 100))
    # shift by +0 since we are zero-based here; original titles mirror indices directly

    # Compute Phi for the selected modes only
    print('Computing selected spatial modes (Phi) for visualization...')
    Phi_sel = compute_phi_selected(files, avg_img, V, evals, mode_indices,
                                   block_size=block_size)

    # Save spatial mode images
    print('Saving snapshots of selected POD modes...')
    save_mode_images(Phi_sel, mode_indices, H, W, avg_img, maskimg,
                     pod_analysis_dir)

    # Optional: write full Phi (as Y_snap = Phi^T) to HDF5
    if write_full_phi:
        print('Writing FULL spatial modes matrix (Phi^T) to HDF5 — this may be VERY large...')
        write_phi(files, avg_img, V, evals,
                  os.path.join(pod_analysis_dir, 'SpecialPOD.hdf5'),
                  block_size=block_size, chunk_cols=phi_chunk_cols)
    else:
        # For compatibility with downstream scripts expecting SpecialPOD.hdf5,
        # we still write an HDF5 with only the selected modes to keep size reasonable.
        with h5py.File(os.path.join(pod_analysis_dir,
                                    'SpecialPOD_selected.hdf5'), 'w') as h5f:
            h5f.create_dataset('data', data=Phi_sel.T, compression='gzip',
                               compression_opts=4)
        print("Wrote SpecialPOD_selected.hdf5 (selected modes only). Set write_full_phi=True to write all modes.")
    # Reconstruct snapshots using modes start_mode..n_modes_995-1 (like original)
    print('Reconstructing snapshots using selected modes...')
    # Ensure start_mode and end index are in bounds
    s = max(0, int(start_mode))
    e = n_modes  # exclusive upper bound
    use_indices = np.array([k for k in range(s, e)], dtype=np.int32)
    Phi_sel1 = compute_phi_selected(files, avg_img, V, evals, use_indices, block_size=block_size)
    
    snr_vals = compute_mode_snr(Phi_sel1, H, W)
    mode_indices = select_modes(use_indices, snr_vals, snr_min=min_snr)
    
    print(np.amin(snr_vals), np.amax(snr_vals), len(use_indices), len(mode_indices))
    
    
    # Safety: fallback to first mode if none survive
    if len(mode_indices) == 0:
        print("⚠️ No modes passed the SNR filter, falling back to mode 0.")
        mode_indices = use_indices
    

    # Compute Phi for the full band of used modes (for reconstruction only)
    Phi_reco = compute_phi_selected(files, avg_img, V, evals, mode_indices, block_size=block_size)

    reconstruct_and_save(Phi_reco, V, mode_indices, avg_img, maskimg,
                         pod_reconstruction_dir, evals, start_idx=s, limit=100)

    # Final time
    dt = time.time() - t0
    if dt > 3600:
        h = int(dt // 3600)
        m = int((dt % 3600) // 60)
        ssec = dt % 60
        print(f"Total run time: {h} Hr, {m} Min, {ssec:.2f} Sec")
    elif dt > 60:
        m = int(dt // 60)
        ssec = dt % 60
        print(f"Total run time: {m} Min, {ssec:.2f} Sec")
    else:
        print(f"Total run time: {dt:.2f} Sec")
