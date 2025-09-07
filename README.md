# Schlieren POD Analysis for Large Image Datasets
This Python script performs a Proper Orthogonal Decomposition (POD) on very large sets of images, such as those from high-speed schlieren. It's specifically optimized to handle datasets that are too large to fit into memory, employing a two-pass streaming approach to manage memory efficiently.

## Introduction:
Proper Orthogonal Decomposition is a data-driven method for extracting dominant, coherent structures from a flow field. Originally developed by Lumley to analyze turbulent flows, POD decomposes a dataset into a set of spatial eigenfunctions (modes) and time-dependent coefficients. These modes are ordered by their energy content, allowing for the identification of the most significant flow structures.

The application of POD to schlieren images is a well-established technique for analyzing density fluctuations in high-speed flows. This approach has been used to examine complex phenomena, such as:

* **Jet Flow Analysis**: Berry et al. (2017) applied snapshot POD to time-resolved schlieren images to extract spatial eigenfunctions and time-dependent coefficients associated with flow structures in a supersonic multi-stream rectangular jet. Their analysis focused on understanding the dominant flow patterns and their temporal evolution.

* **Shock Wave-Boundary Layer Interaction (SBLI)**: Similarly, GUO et al. (2020) utilized schlieren POD to quantify both small-scale pulsations and large-scale fluctuations in the context of SBLI. Their investigation provided insights into the physical properties of a compression corner subjected to an oblique shock wave.

### The Snapshot POD Method
Consistent with these studies, this script's core algorithm is the Snapshot POD method. Rather than solving the computationally expensive eigenvalue problem for the spatial modes directly, it solves a smaller, equivalent problem for the temporal covariance matrix. This approach drastically reduces computational requirements, making it feasible to analyze massive datasets without compromising the integrity of the decomposition.

The method relies on the identity:

$$\Phi=\frac{1}{\sqrt{\lambda_i}}X_fV$$

Where:

* $\Phi$ is the matrix of spatial modes.
* $\lambda_i$ are the eigenvalues of the temporal covariance matrix.
* $X_f$ is the fluctuation matrix (snapshots with the mean subtracted).
* $V$ is the matrix of eigenvectors from the temporal covariance matrix $R=X_fX_f^T$.

The script computes $R$ and its eigendecomposition (`evals`, `V`). It then computes 
$\Phi$ by a series of matrix-vector products, again in a streaming, chunked manner to avoid memory overloads.

## Key Features
* **Memory Efficiency**: Avoids loading the entire image data matrix into RAM. It computes the temporal covariance matrix ($R=X_fX_f^T$) in chunks, making it suitable for datasets with thousands of high-resolution images.

* **Two-Pass Streaming**: The processing is broken down into two main passes over the image data:
  * Pass 1: Computes the mean image.
  * Pass 2: Accumulates the temporal covariance matrix R in a blocked, streaming fashion.

* **Selective Computation**: It can compute and save only a specified number of spatial modes (the POD modes themselves) to save disk space.

* **Reconstruction**: Reconstructs and saves a limited number of snapshots using a user-defined subset of the POD modes.

* **Customization**: Several parameters are configurable, including the number of images to process, the number of modes to save, and memory-related knobs like block_size and data types.

## Requirements
To run this script, you need to have the following Python libraries installed:
* Numpy
* Scipy
* h5py
* opencv-python (cv2)
* matplotlib

To install dependencies:
```bash
pip install numpy h5py opencv-python matplotlib
```

## Usage
1. Configuration: Open the script and modify the `Configuration` section at the top. The most important parameter is `imgs_path`, which should point to your image directory. You may also adjust `images_to_read`, `num_modes_to_save`, and `block_size` based on your dataset size and available memory.
```python
# How many images to use (-1 = all in folder)
images_to_read = -1

# How many spatial modes images to save explicitly
num_modes_to_save = 20
start_mode = 1 

# Path to images
imgs_path = r"raw_images\*.png"

# Directory that contains Avg/00.png (mask image)
# (Automatically inferred from imgs_path; expects Avg/00.png next to the "clean" dir)

# Memory/Performance knobs
block_size = 50            # number of snapshots per block when streaming; 64–256 is typical
use_float32_R = True       # accumulate R in float32 (fast, lighter); switch to False for float64
phi_chunk_cols = 50        # how many modes (columns) to compute at once when writing full Phi
reconst_acum_ratio = 100
min_snr = 0.6
```

2. Run: Execute the script from your terminal:
```bash
python SchlierenPOD_Optimized-v4_CPU.py
```

3. Outputs: The script will create two directories in the same parent folder as your images:
  * PODAnalysis: Contains the energy ratio plot, eigenvalues, eigenvectors, and image files of the spatial modes.

<img width="990" alt="0000" src="https://github.com/user-attachments/assets/3032296e-de4d-4cee-9bea-3a6d24c324fe" />

<img width="330" alt="0000" src="https://github.com/user-attachments/assets/46d3fb78-7d53-457a-9c73-17c9dbf45de7" />
<img width="330" alt="0003" src="https://github.com/user-attachments/assets/c609af4c-b4b9-4ddc-a130-f542c540b46c" />
<img width="330" alt="0048" src="https://github.com/user-attachments/assets/74fc083a-999f-457b-b7b7-82320c4fc42e" />

  * PODReconstruction: Contains the reconstructed snapshots.
<img width="330" alt="R-0001" src="https://github.com/user-attachments/assets/4cd48ad5-3f22-4bc4-98ac-bb6e9536746f" />
<img width="330" alt="R-0005" src="https://github.com/user-attachments/assets/942f0ca2-3444-4952-9328-83a1cb548abd" />
<img width="330" alt="R-0075" src="https://github.com/user-attachments/assets/40baec97-82dc-4802-9df8-347ab89a8533" />
