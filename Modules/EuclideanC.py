import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import pandas as pd
from tqdm import tqdm
np.random.seed(42)



#############################################################################
#############################################################################
#############################################################################

# Posterior predictive checking

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.special import expit
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

def sample_adjacency_matrix(P):
    A = np.random.binomial(1, P)
    A = np.triu(A, k=1)
    A = A + A.T
    return A

def compute_network_statistics(G):
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    degrees = np.array([deg for _, deg in G.degree()])
    stats = {
        'Density': nx.density(G),
        'Transitivity': nx.transitivity(G),
        'Assortativity': nx.degree_assortativity_coefficient(G),
        'Average Degree': degrees.mean(),
        'Degree SD': degrees.std(),
        'Average Path Length': nx.average_shortest_path_length(G),
        'Diameter': nx.diameter(G)
    }
    return stats

def detect_communities(A):
    G = nx.from_numpy_array(A)
    comms = list(greedy_modularity_communities(G))
    return [list(c) for c in comms]

def compute_modularity(A, communities):
    G = nx.from_numpy_array(A)
    return nx.community.modularity(G, communities)

def predictive_check(G, samples_Z, samples_a):
    real_stats = compute_network_statistics(G)
    comms_obs = detect_communities(nx.to_numpy_array(G))
    real_modularity = compute_modularity(nx.to_numpy_array(G), comms_obs)


    #Modularity with latent positions
    Z_CM =  np.mean(samples_Z, axis=0)
    cluster_range = range(2, 12)
    silhouette_scores = []
    for k in cluster_range:
        sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
        labels = sc.fit_predict(Z_CM)
        score = silhouette_score(Z_CM, labels)
        silhouette_scores.append(score)
    best_k = cluster_range[np.argmax(silhouette_scores)]
    sc = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
    labels = sc.fit_predict(Z_CM)
    community = [[i for i, l in enumerate(labels) if l==tag] for tag in np.unique(labels)]
    latent_modularity = compute_modularity(nx.to_numpy_array(G), community)


    stats_list = []
    modularities = []

    num_samples = samples_a.shape[0]
    n = samples_Z.shape[1]

    for l in range(num_samples):
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    P[i, j] = expit(samples_a[l] - 0.5 * np.linalg.norm(samples_Z[l, i, :] - samples_Z[l, j, :])**2)

        A = sample_adjacency_matrix(P)
        G_sample = nx.from_numpy_array(A)
        stats = compute_network_statistics(G_sample)
        stats_list.append(stats)

        comms_sim = detect_communities(A)
        mod = compute_modularity(A, comms_sim)
        modularities.append(mod)

    stats_df = {key: [d[key] for d in stats_list] for key in stats_list[0]}
    stats_df["Modularity"] = modularities

   
    fig, axes = plt.subplots(2, 4, figsize=(24, 10)) 
    axes = axes.flatten()


    selected_stats = list(stats_df.keys())[:8]

    for i, stat in enumerate(selected_stats):
        values = stats_df[stat]
        real_value = real_stats[stat] if stat != "Modularity" else real_modularity
        mean_val = np.mean(values)
        q_low, q_high = np.quantile(values, [0.025, 0.975])

        p_value = np.mean(np.array(values) >= real_value)

        sns.histplot(values, color='gray', bins=30, stat='density', edgecolor='white', ax=axes[i])
        axes[i].axvline(q_low, color='red', linestyle=':', linewidth=1, label='2.5% / 97.5%')
        axes[i].axvline(q_high, color='red', linestyle=':', linewidth=1)
        axes[i].axvline(mean_val, color='blue', linestyle='--', linewidth=1, label='Mean')
        if stat != "Modularity":
            axes[i].axvline(real_value, color='black', linestyle='-', linewidth=2, label='Observed')
        else:
            axes[i].axvline(real_value, color='black', linestyle='-', linewidth=2, label='Observed: Greedy')
            axes[i].axvline(latent_modularity, color='black', linestyle='dashdot', linewidth=2, label='Observed: Latent')
        
        

        axes[i].set_title(f"{stat}\nBayesian p-value: {p_value:.3f}", fontsize = 15)
        axes[i].legend(loc='best')

    plt.suptitle("Posterior prediction checking", fontsize = 30)

    plt.tight_layout()
    plt.show()


def plot_pairwise_sociomatrix(G,samples_Z, samples_a, A_real):
    """
    Visualize pairwise posterior distributions for model.

    Parámetros:
        samples_Z (np.ndarray): Latent positions (num_samples, n, d)
        samples_a (np.ndarray): Parameter (num_samples,)
        samples_b (np.ndarray): Parameter (num_samples,)
        A_real (np.ndarray): Sociomatrix (n, n)
    """
    num_samples, n, _ = samples_Z.shape
    n = len(G.nodes())
    probs_samples = np.zeros((num_samples, n, n))

    for l in range(num_samples):
        for i in range(n):
            for j in range(n):
                if i != j:
                    probs_samples[l, i, j] = expit(samples_a[l] - 0.5 * np.linalg.norm(samples_Z[l, i, :] - samples_Z[l, j, :])**2)

    mean_probs = probs_samples.mean(axis=0)


    fig, axes = plt.subplots(n, n, figsize=(10,10))
    cmap = sns.color_palette("Reds", as_cmap=True)
    
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i > j:
                color = cmap(mean_probs[i, j])
                ax.set_facecolor(color)
                ax.text(0.5, 0.5, f"{mean_probs[i, j]:.3f}", ha='center', va='center', fontsize=10)
            elif i < j:
                vals = probs_samples[:, i, j]
                mean_val = vals.mean()
                real_val = A_real[i, j]
                
                sns.histplot(vals, bins=15, ax=ax, color=cmap(mean_val), edgecolor='white')
                ax.set_xlim(-0.1, 1.1)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label='Mean')
                ax.axvline(real_val, color='black', linestyle='-', linewidth=2.0, label='Real')
            else:
                # Diagonal
                ax.set_facecolor('white')
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()


#############################################################################
#############################################################################
#############################################################################

# Model comparison criteria

def compute_log_likelihood(A_obs, samples_Z, samples_a):
    """
    Compute log-likelihood for each sample and pair of nodes.

    Returns:
        log_lik: np.ndarray de shape (L, n, n)
    """
    L, n, _ = samples_Z.shape
    log_lik = np.zeros((L, n, n))

    for l in range(L):
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_sq = np.linalg.norm(samples_Z[l, i] - samples_Z[l, j])**2
                    p_ij = expit(samples_a[l] - 0.5 * dist_sq)
                    if A_obs[i, j] == 1:
                        log_lik[l, i, j] = np.log(p_ij + 1e-10)
                    else:
                        log_lik[l, i, j] = np.log(1 - p_ij + 1e-10)
    return log_lik

def compute_WAIC(log_lik):
    """
    Compute  WAIC using loglikelihood matrix.
    """
    lppd = np.sum(np.log(np.mean(np.exp(log_lik), axis=0) + 1e-10))  # log pointwise predictive density
    p_waic = np.sum(np.var(log_lik, axis=0))                         # penalization term
    waic = -2 * (lppd - p_waic)
    return waic

def compute_DIC(log_lik, A_obs, samples_Z, samples_a):
    """
    Compute the Deviance Information Criterion (DIC).
    """
    deviance_samples = -2 * np.sum(log_lik, axis=(1, 2))  
    mean_deviance = np.mean(deviance_samples)


    Z_mean = np.mean(samples_Z, axis=0)
    a_mean = np.mean(samples_a)


    n = A_obs.shape[0]
    log_lik_mean = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_sq = np.linalg.norm(Z_mean[i] - Z_mean[j])**2
                p_ij = expit(a_mean - 0.5 * dist_sq)
                if A_obs[i, j] == 1:
                    log_lik_mean[i, j] = np.log(p_ij + 1e-10)
                else:
                    log_lik_mean[i, j] = np.log(1 - p_ij + 1e-10)
    deviance_mean = -2 * np.sum(log_lik_mean)

    p_dic = mean_deviance - deviance_mean
    dic = mean_deviance + p_dic
    return dic

#############################################################################
#############################################################################
#############################################################################

# Prediction evaluation


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.special import expit

# --------------------------------------------------
# Posterior connection probabilities
# --------------------------------------------------
def posterior_connection_probs(samples_Z, samples_a):
    """
    Compute posterior mean probabilities for each pair (i, j)
    based on samples of latent positions Z and intercepts a.
    """
    L, n, _ = samples_Z.shape
    probs = np.zeros((n, n))

    for l in range(L):
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_sq = np.sum((samples_Z[l, i] - samples_Z[l, j]) ** 2)
                    probs[i, j] += expit(samples_a[l] - 0.5 * dist_sq)
    probs /= L
    return probs

# --------------------------------------------------
# ROC, AUC, and manual evaluation metrics
# --------------------------------------------------
def roc_analysis(A_obs, prob_mean, threshold: float = 0.5):
    """
    Plot ROC curve, compute AUC, build confusion matrix manually,
    and report prediction metrics.
    """
    # ---------------------------------------------
    # Prepare ground truth and model probabilities
    # ---------------------------------------------
    tri_mask = np.triu_indices_from(A_obs, k=1)
    y_true = (A_obs[tri_mask] > 0.5).astype(int)  # Ground truth (0 or 1)
    y_score = prob_mean[tri_mask].flatten()       # Posterior mean probabilities

    # ---------------------------------------------
    # ROC Curve & AUC
    # ---------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # ---------------------------------------------
    # Manual confusion matrix
    # ---------------------------------------------
    y_pred = (y_score >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    

    cm = np.array([[tn, fp],
                   [fn, tp]])

    # ---------------------------------------------
    # Prediction metrics
    # ---------------------------------------------
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ---------------------------------------------
    # Plot ROC and Confusion Matrix
    # ---------------------------------------------
    plt.figure(figsize=(6, 5))
    # ROC Curve
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------
    # Return metrics
    # ---------------------------------------------
    return {
        "roc_auc": roc_auc,
        "confusion_matrix": {"tp": tp , "tn": tn, "fp": fp, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity
    }

