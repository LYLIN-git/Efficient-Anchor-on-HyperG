import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "Paper_datasets"


def test_npy_file(file_path: Path):
    print(f"\n--- Testing file: {file_path.name} ---")
    if not file_path.exists():
        print(f"Error: File does not exist at {file_path}")
        return

    try:
        data = np.load(file_path)
        print(f"Successfully loaded {file_path.name} using np.load().")
        print(f"  Shape: {data.shape}")
        print(f"  Data type (dtype): {data.dtype}")
        # flatten() to handle multi-dim arrays
        print(f"  First 5 elements: {data.flatten()[:5]}")
        print(f"  Last 5 elements: {data.flatten()[-5:]}")
        print(
            f"  Unique values (if few): {np.unique(data) if data.size < 200 else 'Too many to list'}")

    except Exception as e:
        print(f"Error loading {file_path.name} using np.load(): {e}")
        try:
            # If np.load fails, it's NOT a .npy file. Maybe it's text?
            # This part is just for diagnostics, not expected for .npy files.
            text_data = np.loadtxt(file_path, dtype=int)
            print(
                f"Successfully loaded {file_path.name} using np.loadtxt(). (WARNING: Not a .npy file then)")
            print(f"  Shape: {text_data.shape}")
            print(f"  Data type (dtype): {text_data.dtype}")
            print(f"  First 5 elements: {text_data.flatten()[:5]}")
        except Exception as e_txt:
            print(f"  Also failed to load as text with np.loadtxt(): {e_txt}")


# Define paths to the files you want to test
zoo_npy_path = DATA_DIR / "zoo.npy"
zoo_label_npy_path = DATA_DIR / "zoo.label.npy"
mushroom_npy_path = DATA_DIR / "mushroom.npy"
mushroom_label_npy_path = DATA_DIR / "mushroom.label.npy"
# Add other .npy files if you have them, e.g., KNN adjacency matrices
zoo_knn_adj_k10_path = DATA_DIR / "zoo_knn_adj_k10.npy"
mushroom_knn_adj_k10_path = DATA_DIR / "mushroom_knn_adj_k10.npy"


if __name__ == "__main__":
    print("Starting .npy file test script.")
    test_npy_file(zoo_npy_path)
    test_npy_file(zoo_label_npy_path)
    test_npy_file(mushroom_npy_path)
    test_npy_file(mushroom_label_npy_path)
    test_npy_file(zoo_knn_adj_k10_path)
    test_npy_file(mushroom_knn_adj_k10_path)
    print("\nFinished .npy file test script.")
