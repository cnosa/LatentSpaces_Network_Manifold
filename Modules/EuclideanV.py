import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import pandas as pd
from tqdm import tqdm
np.random.seed(42)

import plotly.express as px


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")


#############################################################################
#############################################################################
#############################################################################

#2D

#############################################################################
#############################################################################
#############################################################################



def plot_Z_samples_2D(samples_Z):
    samples_Z = np.array(samples_Z)  
    num_points = samples_Z.shape[1]  
    palette = sns.color_palette("hls", num_points)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    centers = [] 

    for i in range(num_points):
        trace = samples_Z[:, i, :]
        mean_point = trace.mean(axis=0)
        centers.append(mean_point)
        sns.scatterplot(
            ax=axes[0],
            x=trace[:, 0],
            y=trace[:, 1],
            color=palette[i],
            alpha=0.1,
            s=20
        )

        sns.scatterplot(
            ax=axes[0],
            x=[mean_point[0]],
            y=[mean_point[1]],
            color=palette[i],
            marker='X',
            s=150,
            edgecolor='black',
            linewidth=0.5,
            label=f"{i}"
        )

        axes[1].scatter(
            mean_point[0],
            mean_point[1],
            color=palette[i],
            s=150,
            marker='X',
            edgecolor='black',
            linewidth=0.5
        )
        axes[1].text(
            mean_point[0] + 0.1,
            mean_point[1] + 0.001,
            f"{i}",
            fontsize=9
        )

    axes[0].set_title("Samples and Centers")
    axes[0].set_xlabel("First dimension")
    axes[0].set_ylabel("Second dimension")
    axes[0].legend(loc="best", fontsize="small")
    axes[0].grid(True)

    axes[1].set_title("Centers Only")
    axes[1].set_xlabel("First dimension")
    axes[1].set_ylabel("Second dimension")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()





def plot_diagnostics(x1,x2,x3):
    iterations1 = np.arange(1, len(x1) + 1)
    data1 = pd.DataFrame({'Iteration': iterations1, 'Values': x1})
    mean1 = np.mean(x1)
    quantiles1 = np.quantile(x1, [0.025, 0.975])

    iterations2 = np.arange(1, len(x2) + 1)
    data2 = pd.DataFrame({'Iteration': iterations2, 'Values': x2})
    mean2 = np.mean(x2)
    quantiles2 = np.quantile(x2, [0.025, 0.975])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 

    # Subplot 1: Acceptance rate
    axs[0].plot(x3, color='black', alpha=0.9, linewidth=2)
    axs[0].axhline(0.80, color='red', linestyle=':', linewidth=2)
    axs[0].axhline(0.60, color='red', linestyle=':', linewidth=2)
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Acceptance rate")
    axs[0].set_title("Acceptance rate over iterations")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].grid(True)

    # Subplot 2: Hamiltonian
    sns.scatterplot(ax=axs[1], x='Iteration', y='Values', data=data1, color='black', alpha=0.3, s=10)
    axs[1].axhline(mean1, color='blue', linestyle='--', linewidth=1, label='Mean')
    axs[1].axhline(quantiles1[0], color='red', linestyle=':', linewidth=1, label='2.5% y 97.5%')
    axs[1].axhline(quantiles1[1], color='red', linestyle=':', linewidth=1)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Hamiltonian")
    axs[1].legend()

    # Subplot 3: LogLikelihood
    sns.scatterplot(ax=axs[2], x='Iteration', y='Values', data=data2, color='black', alpha=0.3, s=10)
    axs[2].axhline(mean2, color='blue', linestyle='--', linewidth=1, label='Mean')
    axs[2].axhline(quantiles2[0], color='red', linestyle=':', linewidth=1, label='2.5% y 97.5%')
    axs[2].axhline(quantiles2[1], color='red', linestyle=':', linewidth=1)
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("LogLikelihood")
    axs[2].legend()

    sns.despine()
    plt.tight_layout()
    plt.show()




def plot_alpha(x):
    iterations = np.arange(1, len(x) + 1)

    mean_x = np.mean(x)
    quantiles_x = np.quantile(x, [0.025, 0.975])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.scatterplot(x=iterations, y=x, color='black', alpha=0.3, s=10, ax=axes[0])
    axes[0].axhline(mean_x, color='blue', linestyle='--', linewidth=1, label='Mean')
    axes[0].axhline(quantiles_x[0], color='red', linestyle=':', linewidth=1, label='2.5% y 97.5%')
    axes[0].axhline(quantiles_x[1], color='red', linestyle=':', linewidth=1)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].set_title("Markov Chain")
    axes[0].legend()
    sns.despine(ax=axes[0])

    sns.histplot(x, bins=30, kde=False, stat='density', color='gray', edgecolor='white', ax=axes[1])
    axes[1].axvline(mean_x, color='blue', linestyle='--', linewidth=1, label='Mean')
    axes[1].axvline(quantiles_x[0], color='red', linestyle=':', linewidth=1, label='2.5% y 97.5%')
    axes[1].axvline(quantiles_x[1], color='red', linestyle=':', linewidth=1)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Marginal distribution")
    axes[1].legend()
    sns.despine(ax=axes[1])

    plt.show()


def clustering2D(G,node_mapping, Z_ML, a_ML, Z_MAP, a_MAP, Z_CM, a_CM):
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(G.nodes)
    Y = nx.to_numpy_array(G, dtype=float)


    Y_ML = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:
                Y_ML[i,j] = expit(a_ML - 0.5 * np.linalg.norm(Z_ML[i] - Z_ML[j])**2)

    Y_CM = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:
                Y_CM[i,j] = expit(a_CM - 0.5 * np.linalg.norm(Z_CM[i] - Z_CM[j])**2)

    

    Y_MAP = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:
                Y_MAP[i,j] = expit(a_MAP - 0.5 * np.linalg.norm(Z_MAP[i] - Z_MAP[j])**2)

    ##############################################################
    cluster_range = range(2, 12)
    silhouette_scores = []


    for k in cluster_range:
        sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
        labels = sc.fit_predict(Z_MAP)
        score = silhouette_score(Z_MAP, labels)
        silhouette_scores.append(score)


    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Cluster validation using Silhouette score")
    plt.grid(True)
    plt.show()


    best_k = cluster_range[np.argmax(silhouette_scores)]
    print(f"Best number of clusters: {best_k}")


    sc = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
    labels = sc.fit_predict(Z_MAP)


    plt.scatter(Z_MAP[:, 0], Z_MAP[:, 1], c=labels, cmap='viridis')
    plt.title(f"Spectral clustering with k={best_k}")
    plt.show()

    ##############################################################
    boundaries = []
    for i in range(1, len(np.sort(labels))):
        if np.sort(labels)[i] != np.sort(labels)[i-1]:
            boundaries.append(i)

    Y_CM_sorted = Y_CM[np.argsort(labels), :][:, np.argsort(labels)]
    Y_MAP_sorted = Y_MAP[np.argsort(labels), :][:, np.argsort(labels)]

    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    ax1 = sns.heatmap(Y[np.argsort(labels), :][:, np.argsort(labels)], annot=False, cmap="Blues", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax1.axhline(boundary, color='black', linewidth=2)
        ax1.axvline(boundary, color='black', linewidth=2)

    plt.title("True sociomatrix matrix")
    plt.subplot(2,2,2)
    ax2 = sns.heatmap(Y_ML[np.argsort(labels), :][:, np.argsort(labels)], annot=False, cmap="Reds", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax2.axhline(boundary, color='black', linewidth=2)
        ax2.axvline(boundary, color='black', linewidth=2)
    plt.title("ML: Estimated sociomatrix matrix")
    plt.subplot(2,2,3)
    ax3 = sns.heatmap(Y_CM_sorted, annot=False, cmap="Reds", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax3.xaxis.set_ticks_position('top')
    ax3.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax3.axhline(boundary, color='black', linewidth=2)
        ax3.axvline(boundary, color='black', linewidth=2)
    plt.title("CM: Estimated sociomatrix matrix")
    plt.subplot(2,2,4)
    ax4 = sns.heatmap(Y_MAP_sorted, annot=False, cmap="Reds", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax4.xaxis.set_ticks_position('top')
    ax4.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax4.axhline(boundary, color='black', linewidth=2)
        ax4.axvline(boundary, color='black', linewidth=2)
    plt.title("MAP: Estimated sociomatrix matrix")
    plt.show()

    ##############################################################

    plt.figure(figsize=(6,6))
    cmap = plt.colormaps.get_cmap('Set1')
    nx.draw(G, nx.spring_layout(G, seed=4),with_labels=False, node_color=[cmap(label) for label in labels], edge_color='gray', node_size=800)
    node_mapping_inv = {v: k for k, v in node_mapping.items()}
    nx.draw_networkx_labels(G, nx.spring_layout(G, seed=4), labels=node_mapping_inv, font_size=14, font_color='black')
    plt.title("Florentine Families Network")
    plt.show()

#############################################################################
#############################################################################
#############################################################################

#3D

#############################################################################
#############################################################################
#############################################################################


def plot_Z_samples_3D(samples_Z):
    samples_Z = np.array(samples_Z)  # shape: (num_samples, num_points, 3)
    num_samples, num_points, _ = samples_Z.shape

    # Flatten the data para hacer scatterplot
    all_points = samples_Z.reshape(-1, 3)
    point_ids = np.tile(np.arange(num_points), num_samples)
    sample_ids = np.repeat(np.arange(num_samples), num_points)

    df = pd.DataFrame(all_points, columns=["x", "y", "z"])
    df["point_id"] = point_ids.astype(str)
    df["sample_id"] = sample_ids

    # Calcular los centros
    centers = samples_Z.mean(axis=0)
    df_centers = pd.DataFrame(centers, columns=["x", "y", "z"])
    df_centers["point_id"] = [str(i) for i in range(num_points)]

    # Graficar las muestras
    fig = px.scatter_3d(
        df,
        x="x", y="y", z="z",
        color="point_id",
        opacity=0.1,
        hover_data=["point_id", "sample_id"]
    )

    # Agregar los centros
    fig.add_scatter3d(
        x=df_centers["x"],
        y=df_centers["y"],
        z=df_centers["z"],
        mode="markers+text",
        marker=dict(size=6, color='black', symbol="x"),
        text=df_centers["point_id"],
        textposition="top center",
        name="Centers"
    )

    fig.update_layout(
        title="Samples and Centers in Latent Space (3D)",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        )
    )

    fig.show()


def clustering3D(G,node_mapping, Z_ML, a_ML, Z_MAP, a_MAP, Z_CM, a_CM):
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(G.nodes)
    Y = nx.to_numpy_array(G, dtype=float)


    Y_ML = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:
                Y_ML[i,j] = expit(a_ML - 0.5 * np.linalg.norm(Z_ML[i] - Z_ML[j])**2)

    Y_CM = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:
                Y_CM[i,j] = expit(a_CM - 0.5 * np.linalg.norm(Z_CM[i] - Z_CM[j])**2)

    

    Y_MAP = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:
                Y_MAP[i,j] = expit(a_MAP - 0.5 * np.linalg.norm(Z_MAP[i] - Z_MAP[j])**2)

    ##############################################################
    cluster_range = range(2, 12)
    silhouette_scores = []


    for k in cluster_range:
        sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
        labels = sc.fit_predict(Z_MAP)
        score = silhouette_score(Z_MAP, labels)
        silhouette_scores.append(score)


    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Cluster validation using Silhouette score")
    plt.grid(True)
    plt.show()


    best_k = cluster_range[np.argmax(silhouette_scores)]
    print(f"Best number of clusters: {best_k}")


    sc = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
    labels = sc.fit_predict(Z_MAP)

    ##############################################################
    boundaries = []
    for i in range(1, len(np.sort(labels))):
        if np.sort(labels)[i] != np.sort(labels)[i-1]:
            boundaries.append(i)

    Y_CM_sorted = Y_CM[np.argsort(labels), :][:, np.argsort(labels)]
    Y_MAP_sorted = Y_MAP[np.argsort(labels), :][:, np.argsort(labels)]

    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    ax1 = sns.heatmap(Y[np.argsort(labels), :][:, np.argsort(labels)], annot=False, cmap="Blues", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax1.axhline(boundary, color='black', linewidth=2)
        ax1.axvline(boundary, color='black', linewidth=2)

    plt.title("True sociomatrix matrix")
    plt.subplot(2,2,2)
    ax2 = sns.heatmap(Y_ML[np.argsort(labels), :][:, np.argsort(labels)], annot=False, cmap="Reds", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax2.axhline(boundary, color='black', linewidth=2)
        ax2.axvline(boundary, color='black', linewidth=2)
    plt.title("ML: Estimated sociomatrix matrix")
    plt.subplot(2,2,3)
    ax3 = sns.heatmap(Y_CM_sorted, annot=False, cmap="Reds", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax3.xaxis.set_ticks_position('top')
    ax3.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax3.axhline(boundary, color='black', linewidth=2)
        ax3.axvline(boundary, color='black', linewidth=2)
    plt.title("CM: Estimated sociomatrix matrix")
    plt.subplot(2,2,4)
    ax4 = sns.heatmap(Y_MAP_sorted, annot=False, cmap="Reds", cbar=False, square=True, linewidths=0.5, linecolor='white',
                xticklabels=np.argsort(labels), yticklabels=np.argsort(labels))
    ax4.xaxis.set_ticks_position('top')
    ax4.xaxis.set_label_position('top')
    for boundary in boundaries:
        ax4.axhline(boundary, color='black', linewidth=2)
        ax4.axvline(boundary, color='black', linewidth=2)
    plt.title("MAP: Estimated sociomatrix matrix")
    plt.show()

    ##############################################################

    plt.figure(figsize=(6,6))
    cmap = plt.colormaps.get_cmap('Set1')
    nx.draw(G, nx.spring_layout(G, seed=4),with_labels=False, node_color=[cmap(label) for label in labels], edge_color='gray', node_size=800)
    node_mapping_inv = {v: k for k, v in node_mapping.items()}
    nx.draw_networkx_labels(G, nx.spring_layout(G, seed=4), labels=node_mapping_inv, font_size=14, font_color='black')
    plt.title("Florentine Families Network")
    plt.show()