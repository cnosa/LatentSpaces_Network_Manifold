import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import pandas as pd
from tqdm import tqdm
import plotly.express as px
np.random.seed(42)



#############################################################################
#############################################################################
#############################################################################

# Loglikelihood function and its gradient for the Euclidean model

def loglikelihood(G,Z,a):
    total = 0.0
    for i in G.nodes():
        for j in G.nodes():
            dist = 0.5 * np.linalg.norm(Z[i] - Z[j])**2
            eta = a - dist
            if j in G.neighbors(i):
                total += eta * 1  + (-np.logaddexp(0, eta))
            elif j != i:
                total += (-np.logaddexp(0, eta))
    return total

def grad_loglikelihood(G,Z,a):
    grad_Z = np.zeros_like(Z)
    grad_a = 0.0
    for i in G.nodes():
        for j in G.nodes():
            if j != i:
                y = 1.0 if j in G.neighbors(i) else 0.0
                dist = 0.5 * np.linalg.norm(Z[i] - Z[j])**2
                eta = a - dist
                grad_Z[i,:] +=  (Z[i] - Z[j]) * (expit(eta) - y)
                grad_a += (-1) * (1) * (expit(eta) - y) 
    return grad_Z, grad_a

#############################################################################
#############################################################################
#############################################################################

# Searching MLE

def SearchingMLE(G, Z_init, a_init, max_iter=1000, tol=1e-4, alpha_init=0.1, rho=0.5, c=1e-4):
    Z0 = Z_init
    historyZ = [Z0]

    a0 = a_init
    historya = [a0]
    
    for i in range(max_iter):
        grad_Z,  grad_a = grad_loglikelihood(G, Z0,a0)
        if np.linalg.norm(grad_Z) + np.abs(grad_a) < tol:
            break  # Convergence criterion
        
        alpha = alpha_init
        
        # Line search using the Armijo condition
        while loglikelihood(G, Z0 + alpha * grad_Z,a0 + alpha*grad_a) < loglikelihood(G, Z0,a0)+ c * alpha * (np.trace(np.transpose(grad_Z) @ Z0) + grad_a * a0):
            alpha *= rho
            if alpha < 1e-9:
                alpha = 0.0
                break
        
        # Update step
        Z0 = Z0 + alpha * grad_Z  
        a0 = a0 + alpha * grad_a
        
        historyZ.append(Z0)
        historya.append(a0)

        if alpha * np.linalg.norm(grad_Z) < tol and alpha * np.abs(grad_a) < tol:
            break
    
    return  Z0, a0, historyZ, historya

#############################################################################
#############################################################################
#############################################################################

# Prior distribution

def logpriori(G,Z,a,var=10):
    n = len(G.nodes)
    return (-1) * (np.log((2*np.pi*var)**((n+1)/2)) + 0.5 * np.sum(Z**2)/var + 0.5 * a**2 / var)

def grad_logpriori(G,Z,a,var=10):
    grad_Z = np.zeros_like(Z)
    for i in G.nodes():
        grad_Z[i,:] = (-1) * Z[i,:] / var
    grad_a = (-1) * a / var  
    return grad_Z, grad_a

#############################################################################
#############################################################################
#############################################################################

# Potential energy function and its gradient

def U(G,Z,a,var=1):
    return (-1) * (loglikelihood(G,Z,a) + logpriori(G,Z,a,var))

def grad_U(G,Z,a,var=1):
    grad_Z_likelihood, grad_a_likelihood = grad_loglikelihood(G,Z,a)
    grad_Z_priori, grad_a_priori = grad_logpriori(G,Z,a,var)
    grad_Z = grad_Z_likelihood + grad_Z_priori
    grad_a = grad_a_likelihood + grad_a_priori
    return -grad_Z, -grad_a



def grad_loglikelihood(G,Z,a):
    grad_Z = np.zeros_like(Z)
    grad_a = 0.0
    for i in G.nodes():
        for j in G.nodes():
            if j != i:
                y = 1.0 if j in G.neighbors(i) else 0.0
                dist = 0.5 * np.linalg.norm(Z[i] - Z[j])**2
                eta = a - dist
                grad_Z[i,:] +=  (Z[i] - Z[j]) * (expit(eta) - y)
                grad_a += (-1) * (1) * (expit(eta) - y) 
    return grad_Z, grad_a


def grad_U_i(G,Z,a,i,var=1):
    grad_Z_i = np.zeros(Z.shape[1])
    grad_a = 0.0
    for j in G.nodes():
        if j != i:
            y = 1.0 if j in G.neighbors(i) else 0.0
            dist = 0.5 * np.linalg.norm(Z[i] - Z[j])**2
            eta = a - dist
            grad_Z_i +=  (Z[i] - Z[j]) * (expit(eta) - y)
            grad_a += (-1) * (1) * (expit(eta) - y) 
    grad_Z_i += (-1) * Z[i,:] / var
    return -grad_Z_i


def grad_U_a(G,Z,a,var=1):
    grad_a = 0.0
    for i in G.nodes():
        for j in G.nodes():
            if j != i:
                y = 1.0 if j in G.neighbors(i) else 0.0
                dist = 0.5 * np.linalg.norm(Z[i] - Z[j])**2
                eta = a - dist
                grad_a += (-1) * (1) * (expit(eta) - y) 
    grad_a += (-1) * a / var 
    return -grad_a

#############################################################################
#############################################################################
#############################################################################

# Applying Hamiltonian Monte Carlo Algorithm


def compute_Z_star(Z, Z0):
    """ Computes Z* = Z0 Z^T (Z Z0^T Z0 Z^T)^(-1/2) Z using SVD """
    A = Z @ Z0.T @ Z0 @ Z.T  # Compute A = Z Z0^T Z0 Z^T
    
    # Compute A^(-1/2) using SVD
    U, S, _ = np.linalg.svd(A)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S))
    A_inv_sqrt = U @ S_inv_sqrt @ U.T
    Z_star =  Z0 @ Z.T @ A_inv_sqrt @ Z
    # Compute Z*
    return Z_star - np.mean(Z_star, axis=0)



def hmc(G, Z_init, a_init, num_samples, epsilon_init=0.05, std_dev_Z=1.0, std_dev_a=1.0, percentage_warmup=0.2, Z0=None):
    """
    Hamiltonian Monte Carlo (HMC) sampling algorithm.
    Parameters:
    - G: Graph object.
    - Z_init: Initial value for Z.
    - a_init: Initial value for a.
    - num_samples: Number of samples to generate.
    - epsilon_init: Initial step size for the leapfrog integrator.
    - std_dev: Standard deviation for the momentum variable.
    - percentage_warmup: Percentage of samples to use for warmup.
    Returns:
    - samples_Z: Generated samples for Z.
    - samples_a: Generated samples for a.
    - Hamiltonian_p: Hamiltonian values for each sample.
    - LogL: Log-likelihood values for each sample.
    - acep_rate_history: Acceptance rate history.
    """

    
    number_of_parameters = Z_init.shape[0] + 1
    warmup = int(num_samples * percentage_warmup)
    number_of_iterations = num_samples + warmup
    print(f"Number of samples: {num_samples}")
    print(f"Number of parameters: {number_of_parameters}")
    print(f"Number of iterations: {number_of_iterations}")
    print(f"Number of warmup iterations: {warmup}")
   

    samples_Z = [Z_init]
    samples_a = [a_init]
    Hamiltonian_p = [U(G,Z_init,a_init)]
    LogL = [loglikelihood(G,Z_init,a_init)]

    acep_rate_history = np.zeros(number_of_iterations)
    

    Z = Z_init.copy()
    a = a_init.copy()

    # Parámetros adaptativos
    epsilon = epsilon_init
    L = max(1, int(round(1/epsilon)))  # L = 1/ε
    accept_count = 0
    total_updates = 0

    
    for iter in tqdm(range(number_of_iterations)):

        # Tunning algorithm parameters
        adapting = iter < warmup
        if adapting and iter > 0:
            current_accept_rate = accept_count / total_updates if total_updates > 0 else 0
            if current_accept_rate < 0.80:
                #epsilon = np.max(np.array([0.05,0.99*epsilon])) 
                std_dev_Z = np.max(np.array([0.05,0.99*std_dev_Z]))
            elif current_accept_rate > 0.60:
                #epsilon = np.min(np.array([0.2,1.01*epsilon]))
                std_dev_Z = np.min(np.array([1.75,1.01*std_dev_Z]))
            L = max(1, int(round(1/epsilon))) 
        elif iter == warmup:
            print(f"Final parameters: epsilon={epsilon:.4f}, L={L}, std_dev_Z={std_dev_Z:.4f}")



        _,  grad_a = grad_U(G,samples_Z[-1],samples_a[-1])


        ### HMC algorithm for Z

        for i in range(Z.shape[0]):
            Z_current = Z.copy()
            Z_i = Z[i].copy()
            grad_Z_i = grad_U_i(G,Z_current, samples_a[-1], i)


            p_i = np.random.normal(0, std_dev_Z, size=Z_i.shape)
            current_p = p_i.copy()

            # Leapfrog integration
            p_i -= epsilon * grad_Z_i / 2
            for _ in range(L):
                Z_i += epsilon * p_i / std_dev_Z**2
                Z_temp = Z_current.copy()
                Z_temp[i] = Z_i  
                grad_Z_i = grad_U_i(G,Z_temp, samples_a[-1], i)
                p_i -= epsilon * grad_Z_i
            p_i -= epsilon * grad_Z_i / 2
  
            # Hamiltonian
            Z_proposed = Z_current.copy()
            Z_proposed[i] = Z_i
            current_U = U(G,Z_current, samples_a[-1])
            proposed_U = U(G,Z_proposed, samples_a[-1])
            current_K = 0.5 * np.sum(current_p**2) / std_dev_Z**2
            proposed_K = 0.5 * np.sum(p_i**2) / std_dev_Z**2
            current_H = current_U + current_K
            proposed_H = proposed_U + proposed_K
            # Metropolis-Hastings acceptance rate
            log_accept_ratio = current_H - proposed_H

            if np.log(np.random.rand()) < log_accept_ratio:
                Z[i] = Z_i 
                accept_count += 1
                samples_Z.append(compute_Z_star(Z.copy(), Z0)) 
                samples_a.append(samples_a[-1])
                Hamiltonian_p.append(proposed_H)
                LogL.append(loglikelihood(G,Z, samples_a[-1]))
            else:
                samples_Z.append(samples_Z[-1])
                samples_a.append(samples_a[-1])
                Hamiltonian_p.append(current_H)
                LogL.append(LogL[-1])

            total_updates += 1
            _,  grad_a = grad_U(G,samples_Z[-1],samples_a[-1])

        ### HMC algorithm for a
        p = np.random.normal(0, std_dev_a, size=1)
        #Leapfrog integration
        p -= epsilon * grad_a / 2        
        for _ in range(L):
            a += epsilon * p / std_dev_a**2
            grad_a = grad_U_a(G,samples_Z[-1],a)
            p -= epsilon * grad_a
        p -= epsilon * grad_a / 2
        # Hamiltonian
        current_H = Hamiltonian_p[-1]
        proposed_U = U(G,samples_Z[-1],a)
        proposed_K = np.sum(p**2) / std_dev_a**2
        proposed_H = proposed_U + proposed_K
        
        # Metropolis-Hastings acceptance rate
        log_accept_ratio = current_H - proposed_H
        if np.log(np.random.rand()) < log_accept_ratio:
            samples_a.append(a.copy())
            samples_Z.append(samples_Z[-1])
            accept_count += 1
            Hamiltonian_p.append(proposed_H)
            LogL.append(loglikelihood(G,samples_Z[-1],a))
        else:
            samples_a.append(samples_a[-1])
            samples_Z.append(samples_Z[-1])
            Hamiltonian_p.append(current_H)
            LogL.append(LogL[-1])
        total_updates += 1    


        acep_rate_history[iter] = accept_count / total_updates if total_updates > 0 else 0

    aceptance_rate = accept_count / total_updates
    print(f"Acceptance rate: {aceptance_rate:.5f}")

    # Choose valid samples
    ## Remove warmup samples
    samples_Z = np.array(samples_Z[1:])[warmup*number_of_parameters:-1:number_of_parameters,:,:]
    samples_a = np.array([np.float64(s.item()) if isinstance(s, np.ndarray) else np.float64(s) for s in samples_a[1:]])[warmup*number_of_parameters:-1:number_of_parameters]
    Hamiltonian_p = np.array([np.float64(s.item()) if isinstance(s, np.ndarray) else np.float64(s) for s in Hamiltonian_p[1:]])[warmup*number_of_parameters:-1:number_of_parameters]
    LogL = np.array([np.float64(s.item()) if isinstance(s, np.ndarray) else np.float64(s) for s in LogL[1:]])[warmup*number_of_parameters:-1:number_of_parameters]
    acep_rate_history = np.array(acep_rate_history)[warmup:]


    return samples_Z, samples_a, Hamiltonian_p, LogL, acep_rate_history

#############################################################################
#############################################################################
#############################################################################

# Saving results


def saving_results(samples, Hamiltonian_p, LogL, acep_rate_history, filename="results.xlsx"):
    samples_Z = samples[0]
    samples_a = samples[1]
    samples_b = samples[2] if len(samples) > 2 else None

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        print("Saving samples...")

        reshaped_Z = samples_Z.reshape(samples_Z.shape[0] * samples_Z.shape[1], samples_Z.shape[2])
        df_Z = pd.DataFrame(reshaped_Z)
        df_Z.to_excel(writer, sheet_name="samples_Z", index=False)

        # samples_a
        df_a = pd.DataFrame(samples_a, columns=["a"])
        df_a.to_excel(writer, sheet_name="samples_a", index=False)

        # samples_b 
        if samples_b is not None:
            samples_b = np.array(samples_b)
            if samples_b.ndim == 1:
                df_b = pd.DataFrame(samples_b, columns=["b"])
            else:
                df_b = pd.DataFrame(samples_b)
            df_b.to_excel(writer, sheet_name="samples_b", index=False)

        print("Saving diagnostics...")

        pd.DataFrame(Hamiltonian_p, columns=["Hamiltonian"]).to_excel(writer, sheet_name="Hamiltonian", index=False)
        pd.DataFrame(LogL, columns=["LogLikelihood"]).to_excel(writer, sheet_name="LogL", index=False)
        pd.DataFrame(acep_rate_history, columns=["AcceptanceRate"]).to_excel(writer, sheet_name="AcceptanceRate", index=False)

    print(f"Results saved to {filename}")