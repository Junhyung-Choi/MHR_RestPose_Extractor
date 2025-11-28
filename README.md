# MHR-RestPose-Extractor

Welcome to the repository for a Python script designed to extract **Canonical Pose (Rest Pose)** and **Bind Pose** data from the [MHR (Momentum Human Rig)](https://github.com/facebookresearch/MHR) model's checkpoint files.

-----

## 🛠️ Usage Guide

### 0\. Clone this Repository

Start by cloning this repository to your local machine:

```bash
git clone https://github.com/Junhyung-Choi/MHR_RestPose_Extractor
cd MHR-RestPose-Extractor
```

### 1\. Download MHR Assets

The necessary model checkpoint file is not included here due to licensing restrictions. You must download the official MHR assets package from the Facebook Research repository.

The required checkpoint file is inside the `assets.zip`.

```bash
# Download the assets package
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
```

### 2\. Unzip the Assets

Unzip the downloaded file to access the checkpoint required by the script.

If you are using a terminal, use the following command:

```bash
unzip assets.zip
```

### 3\. Run the Extraction Script

The provided script contains functions to extract key data (e.g., `rest_pose_vertices`, `rest_pose_joint_rotation`, `skin_weights`, etc.).

**How to Use the Script:**

1.  Open the extraction script (e.g., `extractor.py`).
2.  **Uncomment the function calls** for the specific matrix you want to view or save (e.g., uncomment the line that calls `extract_rest_pose_vertices(...)`).
3.  The extracted data (a PyTorch Tensor) can be passed as an argument to the tensor saving function within the script to save it as a file (e.g., `.npy` or `.pt`).
4.  Execute the script using the path to the downloaded checkpoint file:

<!-- end list -->

```bash
python extractor.py --checkpoint_path assets/mhr_checkpoint.pt
```

-----

## ⚠️ Disclaimer (Data Usage Warning)

The code in this script is free for your use. However, the data extracted using this script (the MHR matrices) is subject to the MHR model's original license.

  * The MHR data extracted via this script is owned by **Facebook Research's MHR project** and is licensed under the **CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)** license.
  * **Commercial use or redistribution of the extracted data is strictly prohibited** by this license. Users must verify the original license and adhere to its non-commercial restriction.

-----

## 🤝 Prerequisites

  * **Python 3.x**
  * **PyTorch** (`pip install torch`)
  * **`curl`** (for downloading the assets)

If you would like me to draft the Python script itself based on these extraction steps, just let me know\!
