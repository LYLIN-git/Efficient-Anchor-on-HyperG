
import diffusion  # C++ diffusion module
from pathlib import Path
import numpy as np
from typing import Sequence, Tuple  # Keep Sequence for type hinting if used
import time
import matplotlib.pyplot as plt  # Added for P_matrix preview plot (optional)




def precompute_anchors(
    graph_path: str,
    k: int,
    lam: float,
    eps: float,
    seed: int,
    out_dir: Path,  # This is ANCHORS_DIR
    T: int,
    overwrite: bool = False,
    cpp_label_path: str = ""
) -> Path:
    np.random.seed(seed)

    # .npz filename now includes T
    anchors_npz_filename = f"P_{Path(graph_path).stem}_k{k}_T{T}.npz"
    anchors_npz = out_dir / anchors_npz_filename
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure ANCHORS_DIR exists

    if anchors_npz.exists() and not overwrite:
        try:
            with np.load(anchors_npz) as data:
                if ('P' in data and
                    'selected_anchors' in data and
                    'precompute_time' in data and
                    data.get('k_val') == k and
                        data.get('T_val') == T):
                    print(
                        f"[anchor_util] using existing anchors: {anchors_npz}", flush=True)
                    return anchors_npz
                else:
                    print(
                        f"[anchor_util] existing {anchors_npz} incomplete or params (k, T) mismatch. Recomputing...", flush=True)
        except Exception as e:
            print(
                f"[anchor_util] Error loading existing {anchors_npz}: {e}. Recomputing...", flush=True)

    print(
        f"[anchor_util] precomputing anchors for {graph_path} (k={k}, T={T})", flush=True)

    temp_solver = diffusion.GraphSolver(
        graph_path, cpp_label_path, "degree", 0)
    n_nodes = temp_solver.n
    if n_nodes == 0:
        raise ValueError(
            f"[anchor_util] Node count is 0 for graph {graph_path}")

    if k > n_nodes:
        print(
            f"WARNING: [anchor_util] k ({k}) > n_nodes ({n_nodes}). Setting k = n_nodes.", flush=True)
        k = n_nodes
    if k == 0:
        print(
            f"WARNING: [anchor_util] k is 0. P_matrix will be empty.", flush=True)
        # Save an empty P with correct first dimension to avoid load errors if query_vector is N-dim
        np.savez(anchors_npz, P=np.zeros((n_nodes, 0), dtype=np.float64), selected_anchors=np.array([]),
                 precompute_time=0.0, k_val=k, T_val=T)
        return anchors_npz

    selected_anchors = np.random.choice(n_nodes, k, replace=False)
    P_matrix = np.zeros((n_nodes, k), dtype=np.float64)
    total_precompute_time = 0.0

    print(
        f"--- [anchor_util] Starting precomputation for {k} anchors ---", flush=True)
    for idx, anchor_node_id in enumerate(selected_anchors):
        print(
            f"    [anchor_util] computing PPR for anchor {idx+1}/{k} (node {anchor_node_id})", flush=True)
        anchor_seed_vector = np.zeros(n_nodes, dtype=np.float64)
        current_anchor_seed_strength = 1.0
        anchor_seed_vector[anchor_node_id] = current_anchor_seed_strength

        print(
            f"    [anchor_util] Python: Anchor seed (node {anchor_node_id}) strength set to: {current_anchor_seed_strength}", flush=True)
        py_norm = np.linalg.norm(anchor_seed_vector)
        print(
            f"    [anchor_util] Python: anchor_seed_vector norm (before reshape): {py_norm:.10e}", flush=True)

        anchor_seed_vector_reshaped = anchor_seed_vector.reshape(-1, 1)

        t0_cpp_call = time.time()
        anchor_diffusion_result_raw = diffusion.diffusion(
            str(graph_path),
            anchor_seed_vector_reshaped,  # Pass N x 1 vector
            T,
            lam,
            eps,
            seed,
            str(cpp_label_path)
        )
        t_elapsed_cpp_call = time.time() - t0_cpp_call
        total_precompute_time += t_elapsed_cpp_call
        print(
            f"    [anchor_util] C++ diffusion call for anchor {idx+1}/{k} done (took {t_elapsed_cpp_call:.2f}s).", flush=True)

        anchor_diffusion_result: Union[np.ndarray, None] = None
        if anchor_diffusion_result_raw is not None and isinstance(anchor_diffusion_result_raw, np.ndarray):
            if anchor_diffusion_result_raw.ndim == 2 and anchor_diffusion_result_raw.shape[1] == 1 and anchor_diffusion_result_raw.shape[0] == n_nodes:
                anchor_diffusion_result = anchor_diffusion_result_raw.ravel()  # Flatten N x 1 to (N,)
                # print(f"    [anchor_util] Flattened C++ result from {anchor_diffusion_result_raw.shape} to {anchor_diffusion_result.shape}", flush=True)
            elif anchor_diffusion_result_raw.ndim == 1 and len(anchor_diffusion_result_raw) == n_nodes:
                anchor_diffusion_result = anchor_diffusion_result_raw
            else:
                print(
                    f"ERROR: [anchor_util] Unexpected shape or length from C++ for anchor {anchor_node_id}: {anchor_diffusion_result_raw.shape if isinstance(anchor_diffusion_result_raw, np.ndarray) else 'Not an ndarray'}", flush=True)
        else:
            print(
                f"ERROR: [anchor_util] C++ diffusion returned None or non-ndarray for anchor {anchor_node_id}", flush=True)

        if anchor_diffusion_result is not None:
            P_matrix[:, idx] = anchor_diffusion_result.astype(np.float64)
            current_col_norm = np.linalg.norm(P_matrix[:, idx])
            print(
                f"    [anchor_util] P_matrix column {idx} (for anchor {anchor_node_id}) norm: {current_col_norm:.4e}", flush=True)
            if current_col_norm > 1e-12 and len(P_matrix[:, idx]) > 0:
                col_data = P_matrix[:, idx]
                print(
                    f"    [anchor_util] P_matrix column {idx} stats: min={np.min(col_data):.2e}, max={np.max(col_data):.2e}, mean={np.mean(col_data):.2e}, std={np.std(col_data):.2e}", flush=True)
                non_zero_count = np.sum(np.abs(col_data) > 1e-9)
                print(
                    f"    [anchor_util] P_matrix column {idx} non-zero elements (approx > 1e-9): {non_zero_count}/{n_nodes} ({(non_zero_count/n_nodes)*100:.2f}%)", flush=True)
        else:
            # Fill with zeros if C++ result was invalid
            P_matrix[:, idx] = np.zeros(n_nodes, dtype=np.float64)
            print(
                f"    [anchor_util] P_matrix column {idx} (for anchor {anchor_node_id}) set to ZEROS due to invalid C++ result.", flush=True)
        print("-" * 20, flush=True)

    print(
        f"\n--- [anchor_util] Finished precomputation for all {k} anchors ---", flush=True)
    if P_matrix.shape[1] > 0:
        column_norms = np.linalg.norm(P_matrix, axis=0)
        print(
            f"[anchor_util] P_matrix ALL column norms (min, max, mean, std): {np.min(column_norms):.2e}, {np.max(column_norms):.2e}, {np.mean(column_norms):.2e}, {np.std(column_norms):.2e}")
        print(
            f"[anchor_util] P_matrix OVERALL (min, max, mean, std): {np.min(P_matrix):.2e}, {np.max(P_matrix):.2e}, {np.mean(P_matrix):.2e}, {np.std(P_matrix):.2e}")
        near_zero_threshold = 1e-9
        num_near_zero_columns = np.sum(
            np.all(np.abs(P_matrix) < near_zero_threshold, axis=0))
        if num_near_zero_columns > 0:
            print(
                f"WARNING: [anchor_util] P_matrix has {num_near_zero_columns} columns where all elements are < {near_zero_threshold}.")
        if np.all(np.abs(P_matrix) < near_zero_threshold):
            print(
                "CRITICAL WARNING: [anchor_util] Entire P_matrix is effectively zero!")
    np.savez(anchors_npz, P=P_matrix, selected_anchors=selected_anchors,
             precompute_time=total_precompute_time, k_val=k, T_val=T)
    print(
        f"[anchor_util] anchors precomputed and saved to {anchors_npz} (total time: {total_precompute_time:.2f}s)", flush=True)
    return anchors_npz


def anchor_infer_least_squares(
    anchors_npz: str, query_vector: np.ndarray, delta: float = 1e-6
) -> Tuple[np.ndarray, float]:
    data_from_npz = np.load(anchors_npz)
    P = data_from_npz["P"]
    if P.size == 0 or P.shape[1] == 0:
        print(
            "ERROR: [LS_Infer] P matrix is empty or has no columns. Cannot perform inference.", flush=True)
        return np.zeros_like(query_vector, dtype=np.float64), np.inf
    n_nodes, k_anchors_from_P = P.shape
    s_query = query_vector.astype(np.float64)

    G = P.T.dot(P)
    b = P.T.dot(s_query)

    delta_actual = delta
    alpha = np.array([], dtype=np.float64)
    if k_anchors_from_P > 0:
        if np.linalg.cond(G) > 1e8:
            scaling_factor = max(1.0, min(np.linalg.cond(G) / 1e8, 1e6))
            delta_actual = delta * scaling_factor
        identity_matrix = np.eye(k_anchors_from_P, dtype=G.dtype)
        try:
            alpha = np.linalg.solve(G + delta_actual * identity_matrix, b)
        except np.linalg.LinAlgError:
            print(
                f"    [LS_Infer] Singular matrix with delta {delta_actual}. Trying pseudo-inverse.", flush=True)
            alpha = np.linalg.pinv(G + delta_actual * identity_matrix).dot(b)

    x_hat = P.dot(alpha) if k_anchors_from_P > 0 else np.zeros_like(
        s_query)  
    loss = float(np.linalg.norm(x_hat - s_query)**2)

    return x_hat.astype(np.float64), loss


def anchor_infer_linear_combo(
    anchors_npz: str, query_vector: np.ndarray
) -> Tuple[np.ndarray, float]:
    data_from_npz = np.load(anchors_npz)
    P = data_from_npz["P"]
    selected_anchor_indices = data_from_npz.get("selected_anchors")
    if P.size == 0 or P.shape[1] == 0 or selected_anchor_indices is None:
        print(
            "ERROR: [LinearCombo_Infer] P or selected_anchors empty/None. Cannot infer.", flush=True)
        return np.zeros_like(query_vector, dtype=np.float64), np.inf
    n_nodes, k_anchors = P.shape
    if len(selected_anchor_indices) != k_anchors:
        print(
            f"ERROR: [LinearCombo_Infer] Mismatch k({k_anchors}) vs selected_anchors({len(selected_anchor_indices)})", flush=True)
        return np.zeros(n_nodes, dtype=np.float64), np.inf
    s_query_at_anchors = query_vector.astype(
        np.float64)[selected_anchor_indices]
    alpha = s_query_at_anchors

    x_hat = P @ alpha if k_anchors > 0 else np.zeros_like(query_vector)

    loss = float(np.linalg.norm(x_hat - query_vector.astype(np.float64))**2)

    return x_hat.astype(np.float64), loss
