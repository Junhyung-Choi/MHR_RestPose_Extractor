import torch
import numpy as np
import os
import argparse
from typing import Dict, Any, List, Union

# ==============================================================================
# 1. File Saving Function
# ==============================================================================

def save_tensor_to_file(data_tensor: Union[torch.Tensor, List[str]], filename: str, out_dir: str = './extracted_data'):
    """
    Saves a torch.Tensor (as a NumPy array) or a list (like joint_names) to a file.
    """
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)
    
    if isinstance(data_tensor, torch.Tensor):
        # Convert PyTorch Tensor to NumPy array and save
        data_array = data_tensor.detach().cpu().numpy()
        np.save(full_path.replace('.npy', ''), data_array)
        print(f"✅ Data saved: {full_path}.npy (Shape: {data_array.shape})")
    elif isinstance(data_tensor, list):
        # Save Python list (e.g., joint names)
        with open(full_path.replace('.txt', '') + '.txt', 'w') as f:
            for item in data_tensor:
                f.write(f"{item}\n")
        print(f"✅ Data saved: {full_path}.txt (Items: {len(data_tensor)})")
    else:
        print(f"⚠️ Warning: Cannot save unsupported data type {type(data_tensor)} for {filename}.")


# ==============================================================================
# 2. Extraction Function
# ==============================================================================

def extract_mhr_data(checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Loads the MHR checkpoint file and extracts the primary LBS matrices and mesh data.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"🔄 Loading file: {checkpoint_path}...")
    
    # Load the TorchScript module
    mhr_model = torch.jit.load(checkpoint_path, map_location=device)
    
    # Access the specific sub-modules as identified by the user
    # Note: Using try/except blocks here is safer for production code, 
    # but based on the user's provided structure, we access directly.
    try:
        char_torch = mhr_model.character_torch
        mesh = char_torch.mesh
        skin = char_torch.linear_blend_skinning
        skel = char_torch.skeleton
    except AttributeError as e:
        print(f"❌ Error accessing expected attributes: {e}")
        print("Please verify the internal structure of the MHR checkpoint file.")
        raise
    
    # Extract data with their expected shapes
    data = {
        # Mesh Structure
        'faces': mesh.faces,                             # (18439, 3) - Triangle indices
        'rest_vertices': mesh.rest_vertices,             # (36874, 3) - Canonical T-pose vertices
        
        # Skinning Data
        'skin_weights_flattened': skin.skin_weights_flattened, # (51337) - Weights (w_j)
        'skin_indices_flattened': skin.skin_indices_flattened, # (51337) - Bone indices (j)
        'vert_indices_flattened': skin.vert_indices_flattened, # (51337) - Vertex indices
        
        # Skeleton Data (Bind Pose components)
        'joint_names': skel.joint_names,                 # List, length: 127
        'joint_prerotations': skel.joint_prerotations,   # (127, 4) - Quaternion pre-rotations for bind pose
        'joint_translation_offsets': skel.joint_translation_offsets, # (127, 3) - Local offsets from parent
        'joint_parents': skel.joint_parents,             # (127) - Parent index for each joint
    }

    print("✅ Data extraction complete.")
    return data

# ==============================================================================
# 3. Main Execution Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="A script to extract LBS matrices from MHR (Momentum Human Rig) checkpoint files.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the MHR checkpoint file (.pt).')
    args = parser.parse_args()

    try:
        extracted_data = extract_mhr_data(args.checkpoint_path)
        
        print("\n=== Extracted Data List ===")
        
        for name, data in extracted_data.items():
            
            # Print shape for tensors, length for lists
            if isinstance(data, torch.Tensor):
                print(f"- {name} (Shape: {data.shape})")
            elif isinstance(data, list):
                 print(f"- {name} (List Length: {len(data)})")
            
            # ------------------------------------------------------------------
            # [User Configuration] 
            # 💡 Uncomment the lines below to save the desired matrices to files.
            # ------------------------------------------------------------------
            
            # if name == 'rest_vertices':
            #     save_tensor_to_file(data, 'rest_vertices.npy')
                
            # if name == 'joint_names':
            #     save_tensor_to_file(data, 'joint_names.txt')

            # if name == 'skin_weights_flattened':
            #     save_tensor_to_file(data, 'skin_weights_flattened.npy')
                
            # if name == 'joint_prerotations':
            #     save_tensor_to_file(data, 'joint_prerotations.npy')

            # if name == 'joint_translation_offsets':
            #     save_tensor_to_file(data, 'joint_translation_offsets.npy')

            # if name == 'faces':
            #     save_tensor_to_file(data, 'faces.npy')
            
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error during data loading or extraction: {e}")

if __name__ == "__main__":
    main()
