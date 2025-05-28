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

def sample_adjacency_matrix(P):
    """
    Sample a binary (0/1) adjacency matrix from a probability matrix P.
    
    Parameters:
        P (np.ndarray): Probability matrix of shape (n, n), with entries in [0,1]
        
    Returns:
        A (np.ndarray): Symmetric adjacency matrix sampled from Bernoulli(P)
    """
    A = np.random.binomial(1, P)

    A = np.triu(A, k=1)
    A = A + A.T
    
    return A

def compute_network_statistics(G):
    """Compute network statistics for a single graph (largest connected component)."""
    
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    degrees = np.array([deg for _, deg in G.degree()])

    stats = {
        'Density': nx.density(G),
        'Transitivity': nx.transitivity(G),
        'Assortativity': nx.degree_assortativity_coefficient(G),
        'Average Degree': degrees.mean(),
        'Degree SD': degrees.std(),
        'Average Path Length': nx.average_shortest_path_length(G)
    }

    return stats



def predictive_check(G, samples_Z, samples_a):
    """
    Perform posterior predictive checks and return summary statistics including Bayesian p-values.
    
    Parameters:
        samples_Z (np.ndarray): Latent position samples (n_samples, n_nodes, d)
        samples_a (np.ndarray): Parameter samples (n_samples,)
        
    Returns:
        df_stats (pd.DataFrame): Sampled network statistics
        bayes_pvals (dict): Bayesian p-values for each statistic
    """

    real_stats = compute_network_statistics(G)
    stats_list = []


    num_samples = samples_a.shape[0]
    
    for l in range(num_samples):
        P = np.zeros((samples_Z.shape[1], samples_Z.shape[1]))
        for i in range(samples_Z.shape[1]):
            for j in range(samples_Z.shape[1]):
                if i != j:
                    P[i, j] = expit(samples_a[l] - 0.5 * np.linalg.norm(samples_Z[l,i,:] - samples_Z[l,j,:])**2)
        
        A = sample_adjacency_matrix(P)
        G_sample = nx.from_numpy_array(A)
        
        stats = compute_network_statistics(G_sample)
        stats_list.append(stats)
    
    df_stats = pd.DataFrame(stats_list)

    bayes_pvals = {}
    for stat_name in df_stats.columns:
        sim_vals = df_stats[stat_name].values
        real_val = real_stats.get(stat_name, None)
        if real_val is not None:
            p_val = np.mean(sim_vals >= real_val)
            bayes_pvals[stat_name] = p_val

    stats_names = df_stats.columns.tolist()

    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, stat_name in enumerate(stats_names):
        if i >= 6:
            break
        
        data = df_stats[stat_name].values
        real_value = real_stats.get(stat_name, None)
        mean_val = np.mean(data)
        q_low, q_high = np.quantile(data, [0.025, 0.975])
        
        ax = axes[i]
        ax.hist(data, bins=25, color='lightgray',edgecolor='white', alpha=0.7, density=True)
        ax.axvline(q_low, color='red', linestyle=':', linewidth=1, label='2.5% y 97.5%')
        ax.axvline(q_high, color='red', linestyle=':', linewidth=1)
        ax.axvline(mean_val, color='blue', linestyle='--', linewidth=1, label='Mean')
        if real_value is not None:
            ax.axvline(real_value, color='black', linestyle='-', linewidth=2, label='True value')
        
        title = f'{stat_name} (p = {bayes_pvals.get(stat_name, np.nan):.4f})'
        ax.set_title(title)
        ax.legend()
    
    for j in range(i+1, 6):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

    #return df_stats, bayes_pvals

def plot_pairwise_sociomatrix(G,samples_Z, samples_a, A_real):
    """
    Visualize pairwise posterior distributions for model.

    Parámetros:
        samples_Z (np.ndarray): Latent positions (num_samples, n, d)
        samples_a (np.ndarray): Parameter (num_samples,)
        A_real (np.ndarray): Sociomatrix (n, n)
    """
    num_samples, n, _ = samples_Z.shape
    n = len(G.nodes())
    probs_samples = np.zeros((num_samples, n, n))

    for l in range(num_samples):
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_sq = np.linalg.norm(samples_Z[l, i] - samples_Z[l, j])**2
                    probs_samples[l, i, j] = expit(samples_a[l] - 0.5 * dist_sq)

    mean_probs = probs_samples.mean(axis=0)


    fig, axes = plt.subplots(n, n, figsize=(10, 10))
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

# Residual analysis

def residual_analysis(A_obs, samples_Z, samples_a):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.special import expit
    from scipy import stats 
    """
    Perform residual analysis comparing observed adjacency to posterior predictive means.
    
    Parameters:
        A_obs (np.ndarray): Observed adjacency matrix (n x n)
        samples_Z (np.ndarray): Latent position samples (L, n, d)
        samples_a (np.ndarray): Samples of intercept (L,)
        
    Returns:
        residuals (np.ndarray): Matrix of standardized Pearson residuals
    """
    n = A_obs.shape[0]
    L = samples_a.shape[0]
    
    # Compute posterior mean of the adjacency matrix
    E_A = np.zeros((n, n))
    for l in range(L):
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_sq = np.linalg.norm(samples_Z[l, i] - samples_Z[l, j])**2
                    p_ij = expit(samples_a[l] - 0.5 * dist_sq)
                    E_A[i, j] += p_ij
    E_A /= L

    # Compute Pearson residuals (with numerical stability)
    epsilon = 1e-6
    residuals = (A_obs - E_A) / np.sqrt(E_A * (1 - E_A) + epsilon)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Heatmap of residuals
    sns.heatmap(A_obs - E_A, ax=axes[0], cmap="coolwarm", center=0, square=True, 
                cbar_kws={'label': 'Residual'})
    axes[0].set_title("Residual Matrix: $Y_{ij} - \mathbb{E}[Y_{ij}]$")

    # Scatter plot of upper triangular residuals
    upper_residuals = residuals[np.triu_indices(n, k=1)]
    axes[1].plot(upper_residuals, color='gray', alpha=0.5, marker='o', linestyle='None')
    axes[1].axhline(upper_residuals.mean(), color='blue', linestyle='--', linewidth=1, label='Mean')

    axes[1].set_title("Pearson residuals")

    # Q-Q plot
    stats.probplot(upper_residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot of Pearson residuals")

    plt.show()

    return residuals  

def residual_distributions(A_obs, samples_Z, samples_a):
    """
    Compute posterior distributions of residuals for each (i,j) pair in the lower triangle.
    
    Parameters:
        A_obs (np.ndarray): Observed adjacency matrix (n x n)
        samples_Z (np.ndarray): Latent positions (L, n, d)
        samples_a (np.ndarray): Intercept samples (L,)
        
    Returns:
        residuals_dict: dict of (i,j) -> list of residual samples
    """
    n = A_obs.shape[0]
    L = samples_a.shape[0]
    residuals_dict = {}

    for i in range(n):
        for j in range(i):
            residual_samples = []
            for l in range(L):
                z_i = samples_Z[l, i]
                z_j = samples_Z[l, j]
                dist_sq = np.sum((z_i - z_j)**2)
                p_ij = expit(samples_a[l] - 0.5 * dist_sq)
                res = A_obs[i, j] - p_ij
                residual_samples.append(res)
            residuals_dict[(i, j)] = np.array(residual_samples)
    
    return residuals_dict

def plot_residual_distributions(residuals_dict, n):
    """
    Plot violinplots or boxplots of residual distributions for each (i,j) in lower triangle.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    data = []
    for (i, j), residuals in residuals_dict.items():
        for r in residuals:
            data.append({'pair': f'{i}-{j}', 'residual': r})

    df = pd.DataFrame(data)

    plt.figure(figsize=(15, 5))
    sns.violinplot(data=df, x='pair', y='residual', inner='quartile', scale='width', palette='coolwarm')
    plt.xticks(rotation=90)
    plt.title("Posterior distributions of residuals per edge (i, j)")
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel("Residual")
    plt.xlabel("Node pair (i-j)")
    plt.tight_layout()
    plt.show()


def summarize_residuals_bayesian(residuals_dict):
    """
    Compute summary diagnostics from posterior distributions of residuals.
    
    Returns:
        df_summary: DataFrame with mean, std, 95% CI, z-score, and bayesian p-value for each (i,j)
    """
    summary = []

    for (i, j), res_samples in residuals_dict.items():
        mean_r = np.mean(res_samples)
        std_r = np.std(res_samples)
        ci_lower, ci_upper = np.percentile(res_samples, [2.5, 97.5])
        z_score = mean_r / (std_r + 1e-8)
        p_value = np.mean(res_samples > 0)

        summary.append({
            "pair": f"{i}-{j}",
            "mean": mean_r,
            "std": std_r,
            "z_score": z_score,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bayes_p": p_value
        })

    df_summary = pd.DataFrame(summary)
    return df_summary

def analyze_residuals_matrix(residuals_dict, n):
    """
    Build matrices of residual summaries: mean, z-score, bayesian p-value, coverage.
    
    Returns:
        dict of summary matrices
    """
    mean_mat = np.full((n, n), np.nan)
    zscore_mat = np.full((n, n), np.nan)
    pval_mat = np.full((n, n), np.nan)
    coverage_mat = np.full((n, n), np.nan)

    for (i, j), samples in residuals_dict.items():
        mean_r = np.mean(samples)
        std_r = np.std(samples) + 1e-6
        z_score = mean_r / std_r
        p_val = np.mean(samples > 0)
        ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
        coverage = int(ci_lower <= 0 <= ci_upper)  # 1 if includes 0, else 0

        mean_mat[i, j] = mean_r
        zscore_mat[i, j] = z_score
        pval_mat[i, j] = p_val
        coverage_mat[i, j] = coverage

    return {
        "mean": mean_mat,
        "zscore": zscore_mat,
        "bayes_p": pval_mat,
        "coverage": coverage_mat
    }

def plot_residual_summary_matrices(summary_matrices):
    """
    Plot residual summary matrices as heatmaps.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
    cmap = "coolwarm"

    sns.heatmap(summary_matrices["mean"], ax=axes[0], cmap=cmap, center=0)
    axes[0].set_title("Mean Residual")

    sns.heatmap(summary_matrices["zscore"], ax=axes[1], cmap=cmap, center=0)
    axes[1].set_title("Z-score")

    sns.heatmap(summary_matrices["bayes_p"], ax=axes[2], cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Bayesian p-value")

    sns.heatmap(summary_matrices["coverage"], ax=axes[3], cmap="Greys", vmin=0, vmax=1, cbar=False)
    axes[3].set_title("95% Credible Interval Covers 0")

    for ax in axes:
        ax.set_xlabel("j")
        ax.set_ylabel("i")
    plt.show()

#############################################################################
#############################################################################
#############################################################################

# Prediction evaluation


from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np

def posterior_connection_probs(samples_Z, samples_a):
    """
    Compute posterior mean probabilities for each (i, j)
    """
    L, n, _ = samples_Z.shape
    probs = np.zeros((n, n))
    
    for l in range(L):
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_sq = np.sum((samples_Z[l, i] - samples_Z[l, j])**2)
                    p_ij = expit(samples_a[l] - 0.5 * dist_sq)
                    probs[i, j] += p_ij
    probs /= L
    return probs

def roc_prc_analysis(A_obs, prob_mean):
    """
    Generate ROC, PRC, AUC, and confusion matrix
    """
    triu_mask = np.triu_indices_from(A_obs, k=1)
    
    # Forzar a binario
    y_true = (A_obs[triu_mask] > 0.5).astype(int)
    y_score = prob_mean[triu_mask].flatten()
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # PRC
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prc_auc = auc(recall, precision)
    
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}",color='blue')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"AUC = {prc_auc:.3f}",color='red')
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    
    # Confusion matrix with threshold 0.5
    y_pred = (y_score > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='coolwarm',square=True, ax=axes[2], cbar=False)
    axes[2].set_title("Confusion matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    return {
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
        "confusion_matrix": cm
    }


#############################################################################
#############################################################################
#############################################################################

# Community analysis

import networkx as nx

   
def simulate_networks(samples_Z, samples_a):
    n = samples_Z.shape[1]
    prob_matrices = []
    for l in range(len(samples_a)):
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    P[i, j] = expit(samples_a[l] - 0.5 * np.linalg.norm(samples_Z[l,i,:] - samples_Z[l,j,:])**2)
        prob_matrices.append(P)

    networks = []
    for P in prob_matrices:
        A_sim = np.random.binomial(1, P)
        np.fill_diagonal(A_sim, 0)
        networks.append(A_sim)
    return networks

def compute_modularity(A, communities):
    G = nx.from_numpy_array(A)
    return nx.community.modularity(G, communities)

from networkx.algorithms.community import greedy_modularity_communities

def detect_communities(A):
    G = nx.from_numpy_array(A)
    comms = list(greedy_modularity_communities(G))
    return [list(c) for c in comms]

def modularity_check(A_obs, samples_Z, samples_a):
    n = A_obs.shape[0]

    prob_matrices = []
    for l in range(len(samples_a)):
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    P[i, j] = expit(samples_a[l] - 0.5 * np.linalg.norm(samples_Z[l,i,:] - samples_Z[l,j,:])**2)
        prob_matrices.append(P)

    A_simulated = simulate_networks(samples_Z, samples_a)
    comms_obs = detect_communities(A_obs)
    mod_obs = compute_modularity(A_obs, comms_obs)

    mod_sim = []
    for A_sim in A_simulated:
        comms_sim = detect_communities(A_sim)
        mod = compute_modularity(A_sim, comms_sim)
        mod_sim.append(mod)

    real_value = mod_obs
    mean_val = np.mean(mod_sim)
    q_low, q_high = np.quantile(mod_sim, [0.025, 0.975])

    sns.histplot(mod_sim, color='gray', bins=30,stat='density', edgecolor='white')
    plt.axvline(q_low, color='red', linestyle=':', linewidth=1, label='2.5% y 97.5%')
    plt.axvline(q_high, color='red', linestyle=':', linewidth=1)
    plt.axvline(mean_val, color='blue', linestyle='--', linewidth=1, label='Mean')
    plt.axvline(mod_obs, color='black', linestyle='--', label=f'Observed modularity = {mod_obs:.6f}')
    plt.title(f"Modularity (p = {np.mean(np.array(mod_sim)>= real_value):.4f})") 
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mod_obs, mod_sim

