import numpy as np
from pathlib import Path

# Adjust paths as needed
# In test_label.py (example path, please verify)
dataset_name = "mushroom"
data_dir = Path("D:/Project/hypergraph_diffusions-main/data/Paper_datasets")
label_npy_path = data_dir / f"{dataset_name}.label.npy"
label_txt_path = data_dir / f"{dataset_name}.label"

# Load .npy labels
try:
    npy_labels = np.load(label_npy_path)
    print(
        f"Loaded {label_npy_path}, shape: {npy_labels.shape}, dtype: {npy_labels.dtype}")
    print(f"First 10 npy_labels: {npy_labels[:10]}")
    print(f"Unique npy_labels: {np.unique(npy_labels)}")
except Exception as e:
    print(f"Error loading {label_npy_path}: {e}")
    npy_labels = None

# Load .txt labels
txt_labels_list = []
try:
    with open(label_txt_path, 'r') as f:
        header = f.readline().strip()  # Skip header if present (e.g., "e p")
        print(
            f"Text label file header: '{header}' (will be skipped if not purely numeric)")
        for line in f:
            try:
                txt_labels_list.append(int(line.strip()))
            except ValueError:
                # If header was not numeric and was read as a label
                if not txt_labels_list and header and not header.isnumeric():  # Only for first problematic line
                    print(
                        f"Skipping non-numeric line assumed to be header: {header}")
                else:
                    print(f"Could not convert line to int: {line.strip()}")
    txt_labels = np.array(txt_labels_list)
    print(
        f"Loaded {label_txt_path}, shape: {txt_labels.shape}, dtype: {txt_labels.dtype}")
    print(f"First 10 txt_labels: {txt_labels[:10]}")
    print(f"Unique txt_labels: {np.unique(txt_labels)}")
except Exception as e:
    print(f"Error loading {label_txt_path}: {e}")
    txt_labels = None

# Comparison
if npy_labels is not None and txt_labels is not None:
    if len(npy_labels) != len(txt_labels):
        print(
            f"CRITICAL MISMATCH: Lengths differ! NPY: {len(npy_labels)}, TXT: {len(txt_labels)}")
    else:
        # Assuming a direct 1-to-1 correspondence in order
        differences = np.sum(npy_labels != txt_labels)
        if differences == 0:
            print(
                "SUCCESS: npy_labels and txt_labels appear to be identical in content and order.")
        else:
            print(
                f"WARNING: Found {differences} differing labels between npy and txt versions.")
            # Find first few differing indices
            diff_indices = np.where(npy_labels != txt_labels)[0]
            print(f"First few differing indices (npy vs txt):")
            for i in diff_indices[:min(5, len(diff_indices))]:
                print(f"  Index {i}: NPY={npy_labels[i]}, TXT={txt_labels[i]}")
