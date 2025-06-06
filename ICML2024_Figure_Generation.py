import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import scipy as sp
from scipy.spatial import distance_matrix

import pdb

from diffusion_functions import *
from semi_supervised_manifold_learning import *

# HELPER FUNCTIONS


def default_rectangle_params(dim_list):
    inner_sidelengths = np.ones(shape=max(dim_list))
    inner_sidelengths[1] = 3
    inner_sidelengths = inner_sidelengths.tolist()
    outer_sidelengths = np.full(shape=max(dim_list), fill_value=2)
    outer_sidelengths[1] = 4
    outer_sidelengths = outer_sidelengths.tolist()
    return inner_sidelengths, outer_sidelengths


def format_axes(ax, titlestring):
    # figure formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axis("off")
    if not titlestring == None:
        ax.set_title(titlestring)
    return


def plot_label_comparison_colorful(ax, label_vector, data_matrix, titlestring=None):
    # make_sweep_cut(label_vector, threshold = 0)
    label_estimates = label_vector

    im = ax.scatter(data_matrix[:, 0], data_matrix[:, 1], c=label_estimates)
    plt.colorbar(im, ax=ax)

    format_axes(ax, titlestring)
    return


def plot_label_comparison_binary(
    ax,
    label_vector,
    data_matrix,
    titlestring=None,
    objective_function=sweep_cut_classification_error,
):

    cut_val, threshold = find_min_sweepcut(
        label_vector, 100, objective_function, orthogonality_constraint="auto"
    )
    label_estimates = make_sweep_cut(label_vector, threshold)

    classification_error = sweep_cut_classification_error(label_estimates)
    orthogonality_error = np.abs(
        np.sum(label_estimates) / len(label_estimates))

    im = ax.scatter(data_matrix[:, 0], data_matrix[:, 1], c=label_estimates)
    plt.colorbar(im, ax=ax)

    subtitle = f"\n Threshold = {threshold:.3f}. Cut objective = {cut_val:.3f} \n Class. error = {classification_error:.3f} \n Orthog. error = {orthogonality_error:.3f}"
    format_axes(ax, titlestring + subtitle)
    return


def graph_vs_hgraph_AUC_hist(AUC_vals, titlestring=None, save=False, folder=None):
    plt.rcParams.update({"font.size": 15})
    # get bin parameters for consistent scaling of both datasets without displaying
    _, first_bins, _ = plt.hist(
        [[v[1] for v in AUC_vals], [v[0] for v in AUC_vals]])
    plt.clf()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(
        [v[1] for v in AUC_vals],
        bins=first_bins,
        alpha=0.5,
        edgecolor="black",
        label="graph",
    )
    ax.hist(
        [v[0] for v in AUC_vals],
        bins=first_bins,
        alpha=0.5,
        edgecolor="black",
        label="hypergraph",
    )

    # figure formatting
    ax.set_title(titlestring)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.legend()
    ax.set_ylabel("Frequency")
    ax.set_xlabel("AUC value")
    if save:
        assert (folder is not None) and (titlestring is not None)
        filename = folder + "/AUC_hist_" + titlestring + ".pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return


def weighted_vs_unweighted_AUC_hist(
    unweighted_AUC=None, weighted_AUC=None, save=False, titlestring=None, folder=None
):
    plt.rcParams.update({"font.size": 15})
    # get bin parameters for consistent scaling of both datasets without displaying
    _, first_bins, _ = plt.hist([unweighted_AUC, weighted_AUC])
    plt.clf()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(
        unweighted_AUC,
        bins=first_bins,
        alpha=0.5,
        color="navy",
        edgecolor="black",
        label="unweighted",
    )
    ax.hist(
        weighted_AUC,
        bins=first_bins,
        alpha=0.5,
        color="firebrick",
        edgecolor="black",
        label="weighted",
    )

    # figure formatting
    ax.set_title(titlestring)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("AUC value")
    # ax.legend()
    if save:
        assert (folder is not None) and (titlestring is not None)
        filename = folder + "/AUC_hist_" + titlestring + ".pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return


def plot_confusion_matrices(cm_vals, titles=None, save=False, titlestring=None, folder=None, suffix=''):
    fig, ax = plt.subplots(2, 1, sharex=True)
    for i, vals in enumerate(cm_vals):
        res = ax[i].imshow(vals, vmin=0, vmax=1,
                           cmap=plt.cm.viridis, interpolation='nearest')
        for x in range(2):
            for y in range(2):
                ax[i].annotate(f'{vals[x][y]:.2f}', xy=(y, x), color='w',
                               horizontalalignment='center',
                               verticalalignment='center')
        if titles is not None and len(titles) == 2:
            ax[i].set_title(titles[i])
        ax[i].set_xticks([0, 1], ['Outer', 'Inner'])
        ax[i].set_yticks([0, 1], ['Outer', 'Inner'], rotation=90)
        ax[i].set_ylabel('Truth')
    ax[1].set_xlabel('Predicted')

    # fig.suptitle(titlestring)
    # cb = fig.colorbar(res)
    if save:
        assert (folder is not None) and (titlestring is not None)
        filename = folder + "/Confusion_Matrix_" + titlestring + suffix + ".png"
        plt.savefig(filename, format="png", bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
    return


# EXPOSITORY FIGURES
def visualize_example_in_2D(type="spheres"):
    print(f"Step1: 開始產生 {type} 資料")
    # generate new data
    if type == "spheres":
        _, data_matrix = generate_concentric_highdim(
            ambient_dim=2, verbose=False)
    elif type == "rectangles":
        # generate rectangles
        inner_sidelengths, outer_sidelengths = default_rectangle_params(dim_list=[
                                                                        2])
        _, data_matrix = generate_concentric_highdim_rectangles(
            inner_sidelengths=inner_sidelengths[:2],
            outer_sidelengths=outer_sidelengths[:2],
            verbose=False,
        )

    print("Step2: 開始標記種子點與產生標籤向量")

    n = data_matrix.shape[0]

    num_rand_seeds = int(0.05 * n)
    x0 = np.full(shape=(n, 1), fill_value=0)
    random_seeds = np.random.choice(
        np.arange(n), size=num_rand_seeds, replace=False)
    assert (
        len(set(random_seeds)) == num_rand_seeds
    ), f"Did not select the right number of seeds. Selected {len(set(random_seeds))} unique seeds instead of {num_rand_seeds}"
    x0[random_seeds[random_seeds < n / 2]] = -1
    x0[random_seeds[random_seeds > n / 2]] = 1

    print("Step3: 開始繪圖與儲存圖像")

    fig, ax = plt.subplots(figsize=(6, 6))
    # formatting
    unlabeled_idxs = (x0 == 0).reshape(
        n,
    )
    plt.scatter(
        data_matrix[unlabeled_idxs, 0],
        data_matrix[unlabeled_idxs, 1],
        marker="x",
        c="grey",
    )
    if type == "spheres":
        # Reversed colorscheme for spheres
        pos_idxs = (x0 == -1).reshape(
            n,
        )
        neg_idxs = (x0 == 1).reshape(
            n,
        )
    elif type == "rectangles":
        pos_idxs = (x0 == 1).reshape(
            n,
        )
        neg_idxs = (x0 == -1).reshape(
            n,
        )

    plt.scatter(data_matrix[pos_idxs, 0],
                data_matrix[pos_idxs, 1], marker="o", c="red")
    plt.scatter(
        data_matrix[neg_idxs, 0], data_matrix[neg_idxs, 1], marker="o", c="blue"
    )
    ax.set_aspect("equal")
    folder = os.path.join("ICML_figs", "examples")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"example_{type}.pdf")
    plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    return


# EXPERIMENTAL FIGURES
def graph_vs_hypergraph_AUC(
    node_weight_method: str,
    manifold_type: str,
    dim_list: list,
    num_trials: int,
    PPR_iterations: int,
    save: bool,
    folder=None,
):
    # Fixed parameters
    pts_per_community = 300  # 設定點 每群 300 點
    k = 5
    order = 2
    # parameters for rectangles
    inner_sidelengths, outer_sidelengths = default_rectangle_params(dim_list)

    if folder is not None:
        os.makedirs(folder, exist_ok=True)

    # Setup problem
    n = 2 * pts_per_community
    labels = np.hstack(
        [
            np.full(shape=int(n / 2), fill_value=-1),
            np.full(shape=int(n / 2), fill_value=1),
        ]
    )

    for ambient_dim in dim_list:
        if manifold_type == "spheres":
            dimension_dependent_data_generation = (
                lambda verbose: generate_concentric_highdim(
                    verbose=False, ambient_dim=ambient_dim
                )
            )
        elif manifold_type == "rectangles":
            dimension_dependent_data_generation = (
                lambda verbose: generate_concentric_highdim_rectangles(
                    verbose=False,
                    inner_sidelengths=inner_sidelengths[:ambient_dim],
                    outer_sidelengths=outer_sidelengths[:ambient_dim],
                )
            )
        AUC_vals = []
        graph_confusion_matrix = np.zeros((2, 2))
        hypergraph_confusion_matrix = np.zeros((2, 2))
        graph_confusion_matrix_balanced = np.zeros((2, 2))
        hypergraph_confusion_matrix_balanced = np.zeros((2, 2))
        for _ in range(num_trials):
            graph_x, hypergraph_x, _ = compare_estimated_labels(
                "PPR",
                generate_data=dimension_dependent_data_generation,
                k=k,
                num_iterations=PPR_iterations,
                diffusion_step_size=None,
                titlestring=None,
                node_weight_method=node_weight_method,
                order=order,
            )

            graph_auc_score = metrics.roc_auc_score(labels, graph_x)
            hypergraph_auc_score = metrics.roc_auc_score(labels, hypergraph_x)

            graph_labels = 2 * (graph_x >= 0) - 1
            graph_confusion_matrix += metrics.confusion_matrix(
                labels, graph_labels) / (num_trials * n) * 2

            graph_argidx = np.argsort(graph_x)
            graph_labels = np.zeros(n)
            graph_labels[graph_argidx[:n//2]] = -1
            graph_labels[graph_argidx[n // 2:]] = 1
            graph_confusion_matrix_balanced += metrics.confusion_matrix(
                labels, graph_labels) / (num_trials * n) * 2

            hypergraph_labels = 2 * (hypergraph_x >= 0) - 1
            hypergraph_confusion_matrix += metrics.confusion_matrix(
                labels, graph_labels) / (num_trials * n) * 2

            hypergraph_argidx = np.argsort(hypergraph_x)
            hypergraph_labels = np.zeros(n)
            hypergraph_labels[hypergraph_argidx[:n // 2]] = -1
            hypergraph_labels[hypergraph_argidx[n // 2:]] = 1
            hypergraph_confusion_matrix_balanced += metrics.confusion_matrix(
                labels, hypergraph_labels) / (num_trials * n) * 2

            AUC_vals.append((hypergraph_auc_score, graph_auc_score))
        if save:
            titlestring = manifold_type + "_dim=" + str(ambient_dim)
        else:
            titlestring = None
        graph_vs_hgraph_AUC_hist(
            AUC_vals, save=save, folder=folder, titlestring=titlestring
        )
        plot_confusion_matrices((graph_confusion_matrix, hypergraph_confusion_matrix), titles=['Graph', 'Hypergraph'],
                                save=save, folder=folder, titlestring=titlestring)
        plot_confusion_matrices((graph_confusion_matrix_balanced, hypergraph_confusion_matrix_balanced), titles=['Graph', 'Hypergraph'],
                                save=save, folder=folder, titlestring=titlestring, suffix='_balanced')

    return


def weighted_vs_unweighted_AUC(
    node_weight_method: str,
    manifold_type: str,
    dim_list: list,
    num_trials: int,
    PPR_iterations: int,
    save: bool,
    folder=None,
    weight_norm_order=2,
):
    # Fixed parameters
    pts_per_community = 300
    k = 5
    order = 2
    # parameters for rectangles
    inner_sidelengths, outer_sidelengths = default_rectangle_params(dim_list)

    if folder is not None:
        os.makedirs(folder, exist_ok=True)

    # Setup problem
    n = 2 * pts_per_community
    labels = np.hstack(
        [
            np.full(shape=int(n / 2), fill_value=-1),
            np.full(shape=int(n / 2), fill_value=1),
        ]
    )

    for ambient_dim in dim_list:
        if manifold_type == "spheres":
            dimension_dependent_data_generation = (
                lambda verbose: generate_concentric_highdim(
                    verbose=False, ambient_dim=ambient_dim
                )
            )
        elif manifold_type == "rectangles":
            dimension_dependent_data_generation = (
                lambda verbose: generate_concentric_highdim_rectangles(
                    verbose=False,
                    inner_sidelengths=inner_sidelengths[:ambient_dim],
                    outer_sidelengths=outer_sidelengths[:ambient_dim],
                )
            )

        unweighted_AUC = []
        weighted_AUC = []
        unweighted_confusion_matrix = np.zeros((2, 2))
        weighted_confusion_matrix = np.zeros((2, 2))
        unweighted_confusion_matrix_balanced = np.zeros((2, 2))
        weighted_confusion_matrix_balanced = np.zeros((2, 2))
        for _ in range(num_trials):
            for node_weight_method, val_list, conf_matrix, conf_matrix_balanced in [
                (None, unweighted_AUC, unweighted_confusion_matrix,
                 unweighted_confusion_matrix_balanced),
                (node_weight_method, weighted_AUC, weighted_confusion_matrix,
                 weighted_confusion_matrix_balanced),
            ]:
                _, hypergraph_x, _ = compare_estimated_labels(
                    "PPR",
                    generate_data=dimension_dependent_data_generation,
                    k=k,
                    num_iterations=PPR_iterations,
                    diffusion_step_size=None,
                    titlestring=None,
                    node_weight_method=node_weight_method,
                    order=weight_norm_order,
                )
                val_list.append(metrics.roc_auc_score(labels, hypergraph_x))
                hypergraph_labels = 2 * (hypergraph_x >= 0) - 1
                conf_matrix += metrics.confusion_matrix(
                    labels, hypergraph_labels) / (num_trials * n) * 2

                hypergraph_argidx = np.argsort(hypergraph_x)
                hypergraph_labels = np.zeros(n)
                hypergraph_labels[hypergraph_argidx[:n // 2]] = -1
                hypergraph_labels[hypergraph_argidx[n // 2:]] = 1
                conf_matrix_balanced += metrics.confusion_matrix(
                    labels, hypergraph_labels) / (num_trials * n) * 2
        if save:
            titlestring = manifold_type + "_dim=" + str(ambient_dim)
        else:
            titlestring = (
                f"Frequency over {num_trials} \n Ambient dimension = {ambient_dim}"
            )
        weighted_vs_unweighted_AUC_hist(
            unweighted_AUC=unweighted_AUC,
            weighted_AUC=weighted_AUC,
            titlestring=titlestring,
            save=save,
            folder=folder,
        )
        plot_confusion_matrices((unweighted_confusion_matrix, weighted_confusion_matrix), titles=['Unweighted', 'Weighted'],
                                save=save, folder=folder, titlestring=titlestring)
        plot_confusion_matrices((unweighted_confusion_matrix_balanced, weighted_confusion_matrix_balanced),
                                titles=['Unweighted', 'Weighted'],
                                save=save, folder=folder, titlestring=titlestring, suffix='balanced')
    return


# Example function calls for recreating the figures in the paper
if __name__ == '__main__':
    print("Step1: 畫 toy 資料分佈：矩形")
    visualize_example_in_2D(type="rectangles")

    print("Step2: 畫 toy 資料分佈：球形")
    visualize_example_in_2D(type="spheres")

    print("Step3: 設定 diffusion 實驗參數")
    num_trials = 50
    PPR_iterations = 50

    graph_vs_hypergraph_AUC(
        node_weight_method="gaussian_to_centroid",
        manifold_type="spheres",
        dim_list=[2, 4, 7, 15],
        num_trials=num_trials,
        PPR_iterations=PPR_iterations,
        save=True,
        folder="./ICML_figs/spheres_gaussian_to_centroid_sigma=2",
    )

    graph_vs_hypergraph_AUC(
        node_weight_method="gaussian_to_centroid",
        manifold_type="rectangles",
        dim_list=[2, 4, 7, 15],
        num_trials=num_trials,
        PPR_iterations=PPR_iterations,
        save=True,
        folder="./ICML_figs/rectangles_gaussian_to_centroid_sigma=2",
    )

    weighted_vs_unweighted_AUC(
        node_weight_method="gaussian_to_centroid",
        manifold_type="spheres",
        dim_list=[2, 15, 30],
        num_trials=num_trials,
        PPR_iterations=PPR_iterations,
        save=True,
        folder="./ICML_figs/weighted_vs_unweighted_spheres_gaussian_to_centroid_sigma=2",
        weight_norm_order=2,
    )

    weighted_vs_unweighted_AUC(
        node_weight_method="gaussian_to_centroid",
        manifold_type="rectangles",
        dim_list=[2, 15, 30],
        num_trials=num_trials,
        PPR_iterations=PPR_iterations,
        save=True,
        folder="./ICML_figs/weighted_vs_unweighted_rectangles_gaussian_to_centroid_sigma=2",
        weight_norm_order=2,
    )
