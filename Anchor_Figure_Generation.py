
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from diffusion import diffusion as cpp_diffusion
from anchor_util import precompute_anchors, anchor_infer_least_squares, anchor_infer_linear_combo
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from typing import Tuple, List, Union

print(">>> LOADED LOCAL Anchor_Figure_Generation.py (with TXT label loading) <<<", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate plots for PPR vs PPR-Anchor.")
    p.add_argument("--dataset", required=True,
                   choices=["zoo", "mushroom", "covertype45", "covertype67", "newsgroups"])
    p.add_argument("--anchor_k", type=int, default=64)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2345)
    p.add_argument("--num_trials", type=int,
                   default=10, help="Number of trials.")
    p.add_argument("--out_dir", type=str,
                   default="Anchor_online_figs", help="Output directory.")
    p.add_argument("--anchor_infer_method", type=str, default="least_squares",
                   choices=["least_squares", "linear_combo"], help="Anchor inference method.")
    return p.parse_args()


def load_labels_from_txt(label_txt_path: Path, num_expected_nodes: int = -1, dataset_name: str = "") -> np.ndarray:
    labels_list = []
    print(
        f"Attempting to load labels from: {label_txt_path} for dataset: {dataset_name}", flush=True)
    try:
        with open(label_txt_path, 'r') as f:

            is_first_line = True
            lines_to_skip = 0

            if dataset_name == "zoo":
                first_line_content = f.readline().strip()
                print(
                    f"DEBUG: Zoo label file, read first line (potential header): '{first_line_content}'", flush=True)

                lines_to_skip = 1

            elif dataset_name == "mushroom":  # Mushroom has 'e p'
                first_line_content = f.readline().strip()
                print(
                    f"DEBUG: Mushroom label file, read first line (potential header): '{first_line_content}'", flush=True)
                try:

                    int(first_line_content.split()[0])

                    if first_line_content.lower() == "e p":
                        lines_to_skip = 1

                except (ValueError, IndexError):
                    lines_to_skip = 1
                if lines_to_skip == 0 and first_line_content:
                    f.seek(0)
                    print(
                        f"DEBUG: Mushroom first line seemed like data, rewinding.", flush=True)

            f.seek(0)
            header_line_content = f.readline().strip()  # Read the first line
            print(
                f"DEBUG: Read (and will skip) header line: '{header_line_content}' from {label_txt_path}", flush=True)

            line_count_data = 0
            for line_content in f:  # Starts from the second line now
                stripped_line = line_content.strip()
                if not stripped_line:
                    continue  # Skip empty lines
                try:
                    labels_list.append(int(stripped_line))
                    line_count_data += 1
                except ValueError:
                    print(
                        f"WARNING: Could not convert line to int in {label_txt_path}: '{stripped_line}'", flush=True)

            if num_expected_nodes > 0 and line_count_data != num_expected_nodes:
                print(
                    f"WARNING: Expected {num_expected_nodes} labels from {label_txt_path} (after skipping header), but found {line_count_data}.", flush=True)

        if not labels_list:  # If list is empty after trying to read
            print(
                f"ERROR: No valid labels loaded from {label_txt_path} (after header skip).", flush=True)
            return np.array([], dtype=int) if num_expected_nodes <= 0 else np.zeros(num_expected_nodes, dtype=int)

        return np.array(labels_list, dtype=int)
    except Exception as e:  # Catch any other read errors
        print(
            f"Error processing text label file {label_txt_path}: {e}", flush=True)
        return np.array([], dtype=int) if num_expected_nodes <= 0 else np.zeros(num_expected_nodes, dtype=int)


def calculate_auc_ovr_multiclass(true_labels: np.ndarray, predicted_scores_all_classes: np.ndarray,
                                 unique_classes: np.ndarray) -> float:

    num_unique_classes = len(unique_classes)
    if num_unique_classes <= 1:
        return 0.5
    if true_labels.shape[0] != predicted_scores_all_classes.shape[0] or predicted_scores_all_classes.shape[1] < num_unique_classes:
        return 0.5
    lb = LabelBinarizer()
    lb.fit(unique_classes)
    try:
        binary_true_labels_ovr = lb.transform(true_labels)
    except ValueError:
        return 0.5

    if binary_true_labels_ovr.ndim == 1 and num_unique_classes == 2:

        binary_true_labels_ovr = binary_true_labels_ovr.reshape(-1, 1)

    if binary_true_labels_ovr.shape[1] == 0:
        return 0.5  # Should not happen if num_unique_classes > 1
    # Anomaly
    if binary_true_labels_ovr.shape[1] == 1 and num_unique_classes > 1 and num_unique_classes != 2:
        print("WARNING: OvR Multiclass LabelBinarizer produced 1 column for >2 unique_classes. Fallback 0.5", flush=True)
        return 0.5

    all_aucs: List[float] = []

    for i in range(binary_true_labels_ovr.shape[1]):
        class_original_index_in_scores = -1
        if num_unique_classes == 2:

            class_original_index_in_scores = lb.classes_[0]

        else:  # Multiclass C > 2
            class_original_index_in_scores = lb.classes_[i]

        current_true_labels_binary = binary_true_labels_ovr[:, i]

        score_column_to_use = -1

        try:

            if class_original_index_in_scores < predicted_scores_all_classes.shape[1]:
                score_column_to_use = class_original_index_in_scores
            else:  # Try to find it if unique_classes were not 0..C-1
                mapped_idx = np.where(
                    unique_classes == class_original_index_in_scores)[0]
                if len(mapped_idx) > 0 and mapped_idx[0] < predicted_scores_all_classes.shape[1]:
                    score_column_to_use = mapped_idx[0]

        except Exception:
            pass

        if score_column_to_use == -1:
            all_aucs.append(0.5)
            continue

        current_predicted_scores = predicted_scores_all_classes[:,
                                                                score_column_to_use]

        if (np.all(current_predicted_scores == current_predicted_scores[0])) or \
           (np.sum(current_true_labels_binary) == 0) or \
           (np.sum(current_true_labels_binary) == len(current_true_labels_binary)):
            all_aucs.append(0.5)
        else:
            try:
                all_aucs.append(roc_auc_score(
                    current_true_labels_binary, current_predicted_scores))
            except ValueError:
                all_aucs.append(0.5)
    if not all_aucs:
        return 0.5
    return np.mean(all_aucs)


def calculate_auc_binary(true_labels_original: np.ndarray, predicted_scores_all_classes: np.ndarray,
                         unique_original_labels: np.ndarray) -> float:
    # ... (Implementation from previous full response, should be okay) ...
    if len(unique_original_labels) != 2:
        return 0.5
    positive_class_label = unique_original_labels[1]
    true_labels_binary_01 = (true_labels_original ==
                             positive_class_label).astype(int)
    scores_for_auc: Union[np.ndarray, None] = None
    if predicted_scores_all_classes.ndim == 1:
        scores_for_auc = predicted_scores_all_classes
    elif predicted_scores_all_classes.shape[1] == 1:
        scores_for_auc = predicted_scores_all_classes.ravel()
    elif predicted_scores_all_classes.shape[1] == 2:

        scores_for_auc = predicted_scores_all_classes[:, 1]
    else:
        return 0.5
    if scores_for_auc is None:
        return 0.5
    if (np.all(scores_for_auc == scores_for_auc[0])) or (np.sum(true_labels_binary_01) == 0) or (np.sum(true_labels_binary_01) == len(true_labels_binary_01)):
        return 0.5
    try:
        auc_val = roc_auc_score(true_labels_binary_01, scores_for_auc)
        return auc_val
    except ValueError:
        return 0.5


def run_ppr(graph_path: str, seeds_indices: np.ndarray, lam: float, eps: float, T: int,
            true_labels_indexed: np.ndarray, unique_original_labels: np.ndarray,
            cpp_label_path: str, current_trial_idx: int, current_revealed_count: int
            ) -> Tuple[np.ndarray, np.ndarray, float]:
    n = true_labels_indexed.shape[0]
    num_classes = len(unique_original_labels)
    X_scores_per_class = np.zeros((n, num_classes), dtype=np.float64)
    total_time_ppr = 0.0
    for c_idx in range(num_classes):  # c_idx is 0 to num_classes-1
        s_c = np.zeros(n, dtype=np.float64)
        for node_idx in seeds_indices:
            if true_labels_indexed[node_idx] == c_idx:
                s_c[node_idx] = lam
            else:
                s_c[node_idx] = -lam
        s_c_reshaped = s_c.reshape(-1, 1)
        t0 = time.time()
        X_c_raw_from_cpp = None
        try:
            X_c_raw_from_cpp = cpp_diffusion(
                str(graph_path), s_c_reshaped, T, lam, eps, 0, str(cpp_label_path))
        except Exception as e:
            X_c_raw_from_cpp = np.zeros((n, 1), dtype=np.float64)
        t_elapsed = time.time() - t0
        total_time_ppr += t_elapsed
        X_c_raw: Union[np.ndarray, None] = None
        if X_c_raw_from_cpp is not None and isinstance(X_c_raw_from_cpp, np.ndarray):
            if X_c_raw_from_cpp.ndim == 2 and X_c_raw_from_cpp.shape[1] == 1 and X_c_raw_from_cpp.shape[0] == n:
                X_c_raw = X_c_raw_from_cpp.ravel()
            elif X_c_raw_from_cpp.ndim == 1 and len(X_c_raw_from_cpp) == n:
                X_c_raw = X_c_raw_from_cpp
            else:
                X_c_raw = np.zeros(n, dtype=np.float64)
        else:
            X_c_raw = np.zeros(n, dtype=np.float64)
        X_scores_per_class[:, c_idx] = X_c_raw
    predicted_labels_indices = np.argmax(X_scores_per_class, axis=1)
    predicted_labels_original = unique_original_labels[predicted_labels_indices]
    return predicted_labels_original, X_scores_per_class, total_time_ppr


def run_anchor(anchors_npz: str, seeds_indices: np.ndarray, lam: float, delta: float,
               true_labels_indexed: np.ndarray, unique_original_labels: np.ndarray,
               current_trial_idx: int, current_revealed_count: int,
               infer_method_name: str
               ) -> Tuple[np.ndarray, np.ndarray, float]:
    n = true_labels_indexed.shape[0]
    num_classes = len(unique_original_labels)
    X_scores_per_class = np.zeros((n, num_classes), dtype=np.float64)
    total_online_time_anchor = 0.0
    for c_idx in range(num_classes):
        s_c_query = np.zeros(n, dtype=np.float64)
        for node_idx in seeds_indices:
            if true_labels_indexed[node_idx] == c_idx:
                s_c_query[node_idx] = lam
            else:
                s_c_query[node_idx] = -lam
        # print(f"    [run_anchor] For class {unique_original_labels[c_idx]}, s_c_query norm: {np.linalg.norm(s_c_query):.2e}", flush=True)
        t0 = time.time()
        x_hat_c: Union[np.ndarray, None] = None
        loss_val = np.nan
        try:
            if infer_method_name == "linear_combo":
                x_hat_c, loss_val = anchor_infer_linear_combo(
                    anchors_npz=str(anchors_npz), query_vector=s_c_query)
            elif infer_method_name == "least_squares":
                x_hat_c, loss_val = anchor_infer_least_squares(
                    anchors_npz=str(anchors_npz), query_vector=s_c_query, delta=delta)
            else:
                raise ValueError(f"Unknown infer_method: {infer_method_name}")
        except Exception as e:
            x_hat_c = np.zeros(n, dtype=np.float64)
        t_elapsed = time.time() - t0
        total_online_time_anchor += t_elapsed
        # print(f"    [run_anchor] {infer_method_name} done (class {unique_original_labels[c_idx]}, t={t_elapsed:.2f}s), loss={loss_val:.4e}", flush=True)
        X_scores_per_class[:, c_idx] = x_hat_c if x_hat_c is not None else np.zeros(
            n, dtype=np.float64)
    predicted_labels_indices = np.argmax(X_scores_per_class, axis=1)
    predicted_labels_original = unique_original_labels[predicted_labels_indices]
    return predicted_labels_original, X_scores_per_class, total_online_time_anchor


if __name__ == "__main__":
    print(">>>> Start Anchor_Figure_Generation (TXT Labels, AUC Strategy 2, AUC Line Plot, Reshape Fix)", flush=True)
    args = parse_args()
    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / "data" / "Paper_datasets"
    ANCHORS_DIR = ROOT / "anchors"
    OUT_DIR = ROOT / args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] out_dir = {OUT_DIR}", flush=True)
    print(
        f"[CONFIG] dataset={args.dataset}, k={args.anchor_k}, T={args.T}, λ={args.lam}, infer={args.anchor_infer_method}", flush=True)

    graph_hmetis_path = DATA_DIR / f"{args.dataset}.hmetis"
    label_txt_path_for_python = DATA_DIR / \
        f"{args.dataset}.label"  # Python uses this now
    cpp_label_path_for_c = DATA_DIR / \
        f"{args.dataset}.label"  # C++ uses this (same file)

    n_nodes_from_graph = 0
    try:
        with open(graph_hmetis_path, 'r') as hf:
            header_line_graph = hf.readline().strip().split()
            if len(header_line_graph) >= 2:
                n_nodes_from_graph = int(header_line_graph[1])
            else:
                raise ValueError("hmetis header too short")
        print(
            f"DEBUG: n_nodes from hmetis header: {n_nodes_from_graph}", flush=True)
    except Exception as e_graph:
        print(
            f"CRITICAL ERROR: Reading n_nodes from {graph_hmetis_path}: {e_graph}. Exiting.")
        exit(1)

    y_true_global = load_labels_from_txt(
        label_txt_path_for_python, n_nodes_from_graph, args.dataset)
    if y_true_global.size == 0 or (n_nodes_from_graph > 0 and y_true_global.size != n_nodes_from_graph):
        print(
            f"CRITICAL ERROR: Label loading failed or mismatch. Loaded {y_true_global.size} labels, expected {n_nodes_from_graph}. Exiting.")
        exit(1)

    unique_labels_global = np.unique(y_true_global)
    num_classes_global = len(unique_labels_global)
    print(
        f"[CONFIG] Loaded {len(y_true_global)} labels. Detected {num_classes_global} unique classes: {unique_labels_global}", flush=True)
    label_to_idx_map = {label: i for i,
                        label in enumerate(unique_labels_global)}
    y_true_indexed = np.array([label_to_idx_map[label]
                              for label in y_true_global], dtype=int)

    all_ppr_aucs_for_hist: List[float] = []
    all_anchor_aucs_for_hist: List[float] = []
    errs_all_trials: List[dict] = []
    times_all_trials: List[dict] = []
    aucs_all_trials: List[dict] = []

    anchors_npz_path = precompute_anchors(str(graph_hmetis_path), args.anchor_k, args.lam,
                                          args.eps, args.seed, ANCHORS_DIR, args.T, False, str(cpp_label_path_for_c))
    print(f"[MAIN] Anchors NPZ: {anchors_npz_path}", flush=True)
    try:
        anchor_npz_data = np.load(anchors_npz_path)
        global_anchor_precompute_time = anchor_npz_data.get(
            "precompute_time", 0.0)
    except FileNotFoundError:
        print(f"ERROR: NPZ not found: {anchors_npz_path}. Exit.")
        exit(1)
    print(
        f"[MAIN] Loaded Anchor Precomp Time: {global_anchor_precompute_time:.2f}s", flush=True)

    for trial_idx in range(args.num_trials):
        print(f"\n>>>> TRIAL {trial_idx+1}/{args.num_trials} <<<<", flush=True)
        np.random.seed(args.seed + trial_idx)
        perm = np.random.permutation(len(y_true_global))
        current_trial_errs_dict = {"ppr": [], "anc": []}
        current_trial_times_dict = {
            "ppr_query_times": [], "anc_online_times": []}
        current_trial_aucs_dict = {"ppr": [], "anc": []}
        reveals_steps = ([20, 25, 30, 35, 40, 45, 50] if args.dataset == "zoo" else [
                         25, 50, 75, 100, 125, 150, 175, 200])
        for r_idx, r_count in enumerate(reveals_steps):
            print(
                f"\n[TRIAL {trial_idx+1}] R={r_count} ({r_idx+1}/{len(reveals_steps)})", flush=True)
            current_revealed_indices = perm[:r_count]
            print("  → PPR …", flush=True)
            ppr_pred_orig, ppr_scores, ppr_time = run_ppr(str(graph_hmetis_path), current_revealed_indices, args.lam,
                                                          args.eps, args.T, y_true_indexed, unique_labels_global, str(cpp_label_path_for_c), trial_idx, r_count)
            ppr_err = 100 * np.mean(ppr_pred_orig != y_true_global)
            ppr_auc = calculate_auc_binary(y_true_global, ppr_scores, unique_labels_global) if num_classes_global == 2 else calculate_auc_ovr_multiclass(
                y_true_global, ppr_scores, unique_labels_global)
            current_trial_errs_dict["ppr"].append(ppr_err)
            current_trial_times_dict["ppr_query_times"].append(ppr_time)
            current_trial_aucs_dict["ppr"].append(ppr_auc)
            print(
                f"  → PPR err={ppr_err:.2f}%, AUC={ppr_auc:.4f}, Time={ppr_time:.2f}s", flush=True)
            print("  → Anchor …", flush=True)
            anc_pred_orig, anc_scores, anc_time = run_anchor(str(
                anchors_npz_path), current_revealed_indices, args.lam, args.delta, y_true_indexed, unique_labels_global, trial_idx, r_count, args.anchor_infer_method)
            anc_err = 100 * np.mean(anc_pred_orig != y_true_global)
            anc_auc = calculate_auc_binary(y_true_global, anc_scores, unique_labels_global) if num_classes_global == 2 else calculate_auc_ovr_multiclass(
                y_true_global, anc_scores, unique_labels_global)
            current_trial_errs_dict["anc"].append(anc_err)
            current_trial_times_dict["anc_online_times"].append(anc_time)
            current_trial_aucs_dict["anc"].append(anc_auc)
            print(
                f"  → Anchor ({args.anchor_infer_method}) err={anc_err:.2f}%, AUC={anc_auc:.4f}, OnlineTime={anc_time:.2f}s", flush=True)
            if r_idx == 0:
                all_ppr_aucs_for_hist.append(ppr_auc)
                all_anchor_aucs_for_hist.append(anc_auc)
        errs_all_trials.append(current_trial_errs_dict)
        times_all_trials.append(current_trial_times_dict)
        aucs_all_trials.append(current_trial_aucs_dict)

    print("\n>>>> Averaging & Plotting…", flush=True)
    avg_errs = {"ppr": np.mean([t["ppr"] for t in errs_all_trials], axis=0), "anc": np.mean(
        [t["anc"] for t in errs_all_trials], axis=0)}
    avg_ppr_query_times = np.mean(
        [t["ppr_query_times"] for t in times_all_trials], axis=0)
    avg_anc_online_times = np.mean(
        [t["anc_online_times"] for t in times_all_trials], axis=0)
    avg_aucs = {"ppr": np.mean([t["ppr"] for t in aucs_all_trials], axis=0), "anc": np.mean(
        [t["anc"] for t in aucs_all_trials], axis=0)}
    std_aucs = {"ppr": np.std([t["ppr"] for t in aucs_all_trials], axis=0), "anc": np.std(
        [t["anc"] for t in aucs_all_trials], axis=0)}
    plot_param_str = f"{args.dataset}_k{args.anchor_k}_T{args.T}_lam{args.lam:.0e}_eps{args.eps:.0e}_delta{args.delta:.0e}_infer{args.anchor_infer_method}_trials{args.num_trials}"

    # Plotting functions (Error, Offline, Online, Original Total, Avg AUC line, AUC Hist)
    # Error Plot
    plt.figure(figsize=(8, 6))
    plt.plot(reveals_steps, avg_errs["ppr"], "-o", label="PPR")
    plt.plot(reveals_steps, avg_errs["anc"], "-s",
             label=f"Anchor ({args.anchor_infer_method})")
    plt.xlabel("# revealed labels")
    plt.ylabel("Error (%)")
    plt.title(f"Avg. Error ({args.dataset}, k={args.anchor_k}, T={args.T})")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"{plot_param_str}_error_avg.png", dpi=300)
    plt.close()
    # Offline Time Plot
    plt.figure(figsize=(7, 5))

    methods_offline = ['Anchor Precomp',
                       f'PPR (Avg. Query r={reveals_steps[0]})']
    ppr_ref_offline = avg_ppr_query_times[0] if len(
        avg_ppr_query_times) > 0 else 0.0
    times_offline_vals = [global_anchor_precompute_time, ppr_ref_offline]

    print(
        f"DEBUG PLOT Offline: global_anchor_precompute_time = {global_anchor_precompute_time}")
    print(f"DEBUG PLOT Offline: avg_ppr_query_times = {avg_ppr_query_times}")
    print(f"DEBUG PLOT Offline: ppr_ref_offline = {ppr_ref_offline}")
    print(f"DEBUG PLOT Offline: times_offline_vals = {times_offline_vals}")
    bars = plt.bar(methods_offline, times_offline_vals,
                   color=['lightcoral', 'skyblue'])

    plt.ylabel("Time (s)")
    plt.title(
        f"Offline Precomp. ({args.dataset}, k={args.anchor_k}, T={args.T})")
    plt.grid(True, axis='y', ls='--', alpha=0.6)
    # Only add text if bars are meaningfully visible
    if global_anchor_precompute_time > 1e-9 or ppr_ref_offline > 1e-9:
        for bar in bars:
            yval = bar.get_height()
            max_val_for_offset = max(times_offline_vals) if max(
                times_offline_vals) > 1e-9 else 1.0
            text_offset = 0.02 * max_val_for_offset
            plt.text(bar.get_x()+bar.get_width()/2., yval +
                     text_offset, f'{yval:.2f}s', ha='center', va='bottom')
    try:
        plt.tight_layout()
    except UserWarning:
        pass
    plt.savefig(
        OUT_DIR/f"{plot_param_str}_offline_precompute_time.png", dpi=300)
    plt.close()
    # Online Time Plot
    plt.figure(figsize=(8, 6))
    plt.plot(reveals_steps, avg_ppr_query_times,
             "-o", label="PPR (Query Time)")
    plt.plot(reveals_steps, avg_anc_online_times, "-s",
             label=f"Anchor ({args.anchor_infer_method}) (Online Time)")
    plt.xlabel("# revealed labels")
    plt.ylabel("Time per Query (s)")
    plt.title(
        f"Avg. Online Inference Time ({args.dataset}, k={args.anchor_k}, T={args.T})")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR/f"{plot_param_str}_online_inference_time_avg.png", dpi=300)
    plt.close()
    # Original Total Time Plot
    old_avg_anc_total_times = []
    for r_i, r_v in enumerate(reveals_steps):
        old_avg_anc_total_times.append((avg_anc_online_times[r_i]+global_anchor_precompute_time) if r_i == 0 and len(
            avg_anc_online_times) > 0 else (avg_anc_online_times[r_i] if len(avg_anc_online_times) > r_i else 0.0))
    plt.figure(figsize=(8, 6))
    plt.plot(reveals_steps, avg_ppr_query_times,
             "-o", label="PPR (Total per Query)")
    plt.plot(reveals_steps, old_avg_anc_total_times, "-s",
             label=f"Anchor ({args.anchor_infer_method}) (Total, Precomp@1st)")
    plt.xlabel("# revealed labels")
    plt.ylabel("Total Runtime (s)")
    plt.title(
        f"Original Avg. Total Runtime ({args.dataset}, k={args.anchor_k}, T={args.T})")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR/f"{plot_param_str}_ORIGINAL_total_runtime_avg.png", dpi=300)
    plt.close()
    # Avg AUC Line Plot
    plt.figure(figsize=(8, 6))
    if len(avg_aucs["ppr"]) == len(reveals_steps) and len(avg_aucs["anc"]) == len(reveals_steps):
        plt.plot(reveals_steps, avg_aucs["ppr"],
                 "-o", label="PPR AUC", color="deepskyblue")
        plt.plot(reveals_steps, avg_aucs["anc"], "-s",
                 label=f"Anchor ({args.anchor_infer_method}) AUC", color="salmon")
        if len(std_aucs["ppr"]) == len(reveals_steps):
            plt.fill_between(reveals_steps, avg_aucs["ppr"]-std_aucs["ppr"],
                             avg_aucs["ppr"]+std_aucs["ppr"], color="deepskyblue", alpha=0.2)
        if len(std_aucs["anc"]) == len(reveals_steps):
            plt.fill_between(reveals_steps, avg_aucs["anc"]-std_aucs["anc"],
                             avg_aucs["anc"]+std_aucs["anc"], color="salmon", alpha=0.2)
    else:
        print("WARN: No/Mismatched data for avg AUC line plot.")
    plt.xlabel("# revealed labels")
    plt.ylabel("Average AUC")
    plt.title(f"Avg. AUC ({args.dataset}, k={args.anchor_k}, T={args.T})")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"{plot_param_str}_auc_avg_line.png", dpi=300)
    plt.close()  # Renamed for clarity
    # AUC Hist Plot
    plt.figure(figsize=(8, 6))
    valid_ppr_hist = [a for a in all_ppr_aucs_for_hist if not np.isnan(a)]
    valid_anc_hist = [a for a in all_anchor_aucs_for_hist if not np.isnan(a)]
    if not valid_ppr_hist and not valid_anc_hist:
        plt.title(
            f"AUC Dist ({args.dataset}, r={reveals_steps[0]}, k={args.anchor_k}, T={args.T}) - NO DATA")
    else:
        data_bins = []
        if valid_ppr_hist:
            data_bins.extend(valid_ppr_hist)
        if valid_anc_hist:
            data_bins.extend(valid_anc_hist)
        bins_h = np.histogram_bin_edges(data_bins, bins=10, range=(
            0.0, 1.0)) if data_bins else np.linspace(0, 1, 11)  # Ensure bins cover 0-1
        if valid_ppr_hist:
            plt.hist(valid_ppr_hist, bins=bins_h, alpha=0.7,
                     label='PPR', color='skyblue', edgecolor='black')
        if valid_anc_hist:
            plt.hist(valid_anc_hist, bins=bins_h, alpha=0.7,
                     label=f"Anchor ({args.anchor_infer_method})", color='lightcoral', edgecolor='black')
        plt.title(
            f"AUC Distribution ({args.dataset}, r={reveals_steps[0]}, k={args.anchor_k}, T={args.T})")
    plt.xlabel("AUC Value")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"{plot_param_str}_auc_histogram.png", dpi=300)
    plt.close()

    print(f"\n▶ All figures saved to {OUT_DIR}", flush=True)
