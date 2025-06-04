import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import plotly.express as px
import pandas as pd
from tqdm import tqdm
np.random.seed(42)



#############################################################################
#############################################################################
#############################################################################

# Loglikelihood function and its gradient for the Euclidean model

def loglikelihood(G,Z,a,b):
    total = 0.0
    for i in G.nodes():
        for j in G.nodes():
            dist = Z[i].T @ Z[j]
            eta = a + b*dist
            if j in G.neighbors(i):
                total += eta * 1  + (-np.logaddexp(0, eta))
            elif j != i:
                total += (-np.logaddexp(0, eta))
    return total

def grad_loglikelihood(G,Z,a,b):
    grad_Z = np.zeros_like(Z)
    grad_a = 0.0
    grad_b = 0.0
    for i in G.nodes():
        for j in G.nodes():
            if j != i:
                y = 1.0 if j in G.neighbors(i) else 0.0
                dist = Z[i].T @ Z[j] 
                eta = a + b*dist
                grad_Z[i,:] +=  (y-expit(eta)) *  (b*Z[j])
                grad_a += (y-expit(eta)) * (1) 
                grad_b += (y-expit(eta)) * (dist) 
    return grad_Z, grad_a, grad_b

def update_Z(Z, grad_Z):
    for i in range(len(Z)):
        proj_orth = grad_Z[i]-np.dot(Z[i], grad_Z[i]) * Z[i]
        Z[i] = Z[i] + proj_orth 
        Z[i] = Z[i] / np.linalg.norm(Z[i])
    return Z
#############################################################################
#############################################################################
#############################################################################

# Searching MLE

def SearchingMLE(G, Z_init, a_init, b_init, max_iter=1000, tol=1e-10, alpha_init=0.1, rho=0.5, c=1e-4):
    Z0 = Z_init
    historyZ = [Z0]

    a0 = a_init
    historya = [a0]

    b0 = b_init
    historyb = [b0]
    
    for i in range(max_iter):
        grad_Z,  grad_a, grad_b = grad_loglikelihood(G, Z0,a0,b0)
        if np.linalg.norm(grad_Z) + np.abs(grad_a) + np.abs(grad_b) < tol:
            break  # Convergence criterion
        
        alpha = alpha_init
        
        # Line search using the Armijo condition
        while loglikelihood(G, update_Z(Z0, alpha*grad_Z) ,a0 + alpha*grad_a,b0 + alpha*grad_b) < loglikelihood(G, Z0,a0,b0)+ c * alpha * (np.trace(np.transpose(grad_Z) @ Z0) + grad_a * a0 + grad_b * b0):
            alpha *= rho
            if alpha < 1e-4:
                alpha = 0.0
                break
        
        # Update step
        Z0 = update_Z(Z0, alpha*grad_Z) 
        a0 = a0 + alpha * grad_a
        b0 = b0 + alpha * grad_b
        
        historyZ.append(Z0)
        historya.append(a0)
        historyb.append(b0)

        #if alpha * np.linalg.norm(grad_Z) < tol and alpha * np.abs(grad_a) < tol:
        #    break
    
    return  Z0, a0, b0, historyZ, historya, historyb

#############################################################################
#############################################################################
#############################################################################

# Prior distribution

def logpriori(G,Z,a,b,var=1.0):
    n = len(G.nodes())
    logpriorZ = 0.0
    for i in range(len(Z)):
        logpriorZ += - (1.0 - Z[i].T @ Z[i])**2 / var
    logpriora = - 0.5 * a**2 / var
    logpriorb = - 0.5 * (b-1)**2 / var
    return np.log((2*np.pi*var)**((n+1)/2)) + logpriorZ + logpriora + logpriorb


def grad_logpriori(G, Z,a,b,var=0.1):
    grad_Z = np.zeros_like(Z)
    for i in G.nodes():
        grad_Z[i,:] = (-1) * Z[i] * (1.0 - Z[i].T @ Z[i])
    grad_a = (-1) * a / var  
    grad_b = (-1) * (b-1) / var  
    return grad_Z, grad_a, grad_b

#############################################################################
#############################################################################
#############################################################################

# Potential energy function and its gradient

def U(G, Z,a,b,var=1):
    return (-1) * (loglikelihood(G, Z,a,b) + logpriori(G, Z,a,b,var))

def grad_U(G, Z,a,b,var=0.1):
    grad_Z_likelihood, grad_a_likelihood, grad_b_likelihood = grad_loglikelihood(G, Z,a,b)
    grad_Z_priori, grad_a_priori, grad_b_priori = grad_logpriori(G, Z,a,b,var)
    grad_Z = grad_Z_likelihood + grad_Z_priori 
    grad_a = grad_a_likelihood + grad_a_priori
    grad_b = grad_b_likelihood + grad_b_priori
    return -grad_Z, -grad_a, -grad_b

def grad_U_i(G,Z,a,b,i):
    var_z = 0.01
    grad_Z_i = np.zeros(Z.shape[1])
    for j in G.nodes():
        if j != i:
            y = 1.0 if j in G.neighbors(i) else 0.0
            dist = Z[i].T @ Z[j] 
            eta = a + b*dist
            grad_Z_i +=  (y-expit(eta)) *  (b*Z[j])
    grad_Z_i += (-1) * Z[i] * (1.0 - Z[i].T @ Z[i]) / var_z
    return -grad_Z_i

def grad_U_a(G,Z,a,b,var=1.0):
    grad_a = 0.0
    for i in G.nodes():
        for j in G.nodes():
            if j != i:
                y = 1.0 if j in G.neighbors(i) else 0.0
                dist = Z[i].T @ Z[j]
                eta = a + b*dist
                grad_a += (y-expit(eta))
    grad_a += (-1) * a / var 
    return -grad_a

def grad_U_b(G,Z,a,b,var=1.0):
    grad_b = 0.0
    for i in G.nodes():
        for j in G.nodes():
            if j != i:
                y = 1.0 if j in G.neighbors(i) else 0.0
                dist = Z[i].T @ Z[j]
                eta = a + b*dist
                grad_b += (y-expit(eta)) * (dist)
    grad_b += (-1) * b / var 
    return -grad_b

#############################################################################
#############################################################################
#############################################################################

# Applying Hamiltonian Monte Carlo Algorithm


def project_to_tangent_space(theta, phi):
    return phi - np.dot(phi, theta) * theta
def geodesic_flow(theta, phi, step_size):
    alpha = np.linalg.norm(phi)
    if alpha > 1e-10:
        new_theta = theta * np.cos(alpha * step_size) + (phi / alpha) * np.sin(alpha * step_size)
        new_phi = phi * np.cos(alpha * step_size) - alpha * theta * np.sin(alpha * step_size)
    else:
        new_theta, new_phi = theta, phi  
    return new_theta, new_phi
def compute_starS1(Old, Ref):
    ang_old = np.arctan2(Old[:,1], Old[:,0])
    ang_ref = np.arctan2(Ref[:,1], Ref[:,0])
    
    addition = np.mean(ang_ref-ang_old)
    
    ang_new = ang_old + addition
    New = np.zeros_like(Old)
    New[:,0] = np.cos(ang_new)
    New[:,1] = np.sin(ang_new)
    return New 


def compute_starS2(Old, Ref):
    inclination_ang_old = np.arctan2(Old[:,1], Old[:,0])
    azimutal_ang_old = np.arccos(Old[:,2])
    inclination_ang_ref = np.arctan2(Ref[:,1], Ref[:,0])
    azimutal_ang_ref = np.arccos(Ref[:,2])
    
    
    addition_inclination = np.mean(inclination_ang_ref - inclination_ang_old)
    addition_inclination = np.mean(azimutal_ang_ref - azimutal_ang_old)
    
    inclination_ang_new = inclination_ang_old + addition_inclination
    azimutal_ang_new = azimutal_ang_old + addition_inclination
    New = np.zeros_like(Old)
    New[:,0] = np.cos(azimutal_ang_new) * np.sin(inclination_ang_new)
    New[:,1] = np.sin(azimutal_ang_new) * np.sin(inclination_ang_new)
    New[:,2] = np.cos(inclination_ang_new)
    return New 


def ghmc(G, Z_init, a_init, b_init, num_samples, epsilon_init=0.05, std_dev_init_Z=0.2, std_dev_init_a = 0.4, std_dev_init_b = 0.2, percentage_warmup=0.2):
    """
    Hamiltonian Monte Carlo (HMC) sampling algorithm.
    Parameters:
    - G: Graph object.
    - Z_init: Initial value for Z.
    - a_init: Initial value for a.
    - num_samples: Number of samples to generate.
    - epsilon_init: Initial step size for the leapfrog integrator.
    - std_dev_init: Standard deviation for the momentum variable.
    - percentage_warmup: Percentage of samples to use for warmup.
    Returns:
    - samples_Z: Generated samples for Z.
    - samples_a: Generated samples for a.
    - Hamiltonian_p: Hamiltonian values for each sample.
    - LogL: Log-likelihood values for each sample.
    - acep_rate_history: Acceptance rate history.
    """

    n = len(G.nodes())
    number_of_parameters = Z_init.shape[0] + 2
    warmup = int(num_samples * percentage_warmup)
    number_of_iterations = num_samples + warmup
    print(f"Number of samples: {num_samples}")
    print(f"Number of parameters: {number_of_parameters}")
    print(f"Number of iterations: {number_of_iterations}")
    print(f"Number of warmup iterations: {warmup}")
   

    samples_Z = [Z_init]
    samples_a = [a_init]
    samples_b = [b_init]
    Hamiltonian_p = [U(G,Z_init,a_init,b_init)]
    LogL = [loglikelihood(G,Z_init,a_init,b_init)]
    # Parámetros adaptativos
    epsilon = epsilon_init
    std_dev_Z = std_dev_init_Z
    std_dev_a = std_dev_init_a
    std_dev_b = std_dev_init_b
    L = max(1, int(round(1/epsilon)))  # L = 1/ε
    accept_count = 0
    total_updates = 0

    acep_rate_history = np.zeros(number_of_iterations)
    
    for iter in tqdm(range(number_of_iterations)):

        Z = samples_Z[-1].copy()
        a = samples_a[-1].copy()
        b = samples_b[-1].copy()


        # Tunning algorithm parameters
        adapting = iter < warmup
        if adapting and iter > 0:
            current_accept_rate = accept_count / total_updates if total_updates > 0 else 0
            if current_accept_rate < 0.80:
                #epsilon = np.max(np.array([0.010,0.99*epsilon])) 
                std_dev_Z = np.max(np.array([0.025,0.99*std_dev_Z]))
            elif current_accept_rate > 0.60:
                #epsilon = np.min(np.array([0.2,1.01*epsilon]))
                std_dev_Z = np.min(np.array([0.75,1.01*std_dev_Z]))
            L = max(1, int(round(1/epsilon)))  
        elif iter == warmup:
            print(f"Final parameters: epsilon={epsilon:.4f}, L={L}, std_dev_Z={std_dev_Z:.4f}, std_dev_a={std_dev_a:.4f}, std_dev_b={std_dev_b:.4f}")



        


        ### GHMC algorithm for Z

        for i in range(Z.shape[0]):
            Z_i = Z[i].copy()
            p_i = np.random.normal(0, std_dev_Z, size=Z_i.shape)
            p_i = project_to_tangent_space(Z_i, p_i)
            current_p = p_i.copy()

            #Leapfrog integration
            grad_Z_i = grad_U_i(G,Z, samples_a[-1], samples_b[-1], i)
            Z_i = Z[i].copy()
            p_i -= epsilon * grad_Z_i / 2 
            p_i = project_to_tangent_space(Z_i, p_i)       
            for _ in range(L):
                Z_i, p_i = geodesic_flow(Z_i, p_i, epsilon)
            Z[i] = Z_i.copy()
            grad_Z_i = grad_U_i(G,Z, samples_a[-1], samples_b[-1], i)
            p_i -= epsilon * grad_Z_i / 2
            p_i = project_to_tangent_space(Z_i, p_i)

            

            # Hamiltonian
            current_U = U(G, samples_Z[-1],samples_a[-1],samples_b[-1])
            current_K = 0.5 * np.sum(current_p**2)
            current_H = current_U + current_K
            proposed_U = U(G, Z,samples_a[-1],samples_b[-1])
            proposed_K = 0.5 * np.sum(p_i**2)
            proposed_H = proposed_U + proposed_K
            # Metropolis-Hastings acceptance rate
            log_accept_ratio = current_H - proposed_H
            if np.log(np.random.rand()) < log_accept_ratio:
                samples_a.append(samples_a[-1])
                samples_b.append(samples_b[-1])
                samples_Z.append(Z.copy())
                accept_count += 1
                Hamiltonian_p.append(proposed_H)
                LogL.append(loglikelihood(G, Z,samples_a[-1],samples_b[-1]))
            else:
                samples_a.append(samples_a[-1])
                samples_b.append(samples_b[-1])
                samples_Z.append(samples_Z[-1])
                Hamiltonian_p.append(current_H)
                LogL.append(LogL[-1])
            total_updates += 1   


        

        ### HMC algorithm for a
        p = np.random.normal(0, std_dev_a, size=1)
        #Leapfrog integration
        grad_a = grad_U_a(G,samples_Z[-1],a,samples_b[-1])
        p -= epsilon * grad_a / 2        
        for _ in range(L):
            a += epsilon * p / std_dev_a**2
            grad_a = grad_U_a(G,samples_Z[-1],a,samples_b[-1])
            p -= epsilon * grad_a
        p -= epsilon * grad_a / 2
        # Hamiltonian
        current_H = Hamiltonian_p[-1]
        proposed_U = U(G,samples_Z[-1],a,samples_b[-1])
        proposed_K = np.sum(p**2) / std_dev_a**2
        proposed_H = proposed_U + proposed_K
        
        # Metropolis-Hastings acceptance rate
        log_accept_ratio = current_H - proposed_H
        if np.log(np.random.rand()) < log_accept_ratio:
            samples_a.append(a.copy())
            samples_b.append(samples_b[-1])
            samples_Z.append(samples_Z[-1])
            accept_count += 1
            Hamiltonian_p.append(proposed_H)
            LogL.append(loglikelihood(G,samples_Z[-1],a,samples_b[-1]))
        else:
            samples_a.append(samples_a[-1])
            samples_b.append(samples_b[-1])
            samples_Z.append(samples_Z[-1])
            Hamiltonian_p.append(current_H)
            LogL.append(LogL[-1])
        total_updates += 1 

        ### HMC algorithm for b
        grad_b = grad_U_b(G,samples_Z[-1],samples_a[-1],b)
        p = np.random.normal(0, std_dev_b, size=1)
        #Leapfrog integration
        p -= epsilon * grad_b / 2        
        for _ in range(L):
            b += epsilon * p / std_dev_b**2
            grad_b = grad_U_b(G,samples_Z[-1],samples_a[-1],b)
            p -= epsilon * grad_b
        p -= epsilon * grad_b / 2
        # Hamiltonian
        current_H = Hamiltonian_p[-1]
        proposed_U = U(G,samples_Z[-1],samples_a[-1],b)
        proposed_K = np.sum(p**2) / std_dev_b**2
        proposed_H = proposed_U + proposed_K
        
        # Metropolis-Hastings acceptance rate
        log_accept_ratio = current_H - proposed_H
        if np.log(np.random.rand()) < log_accept_ratio:
            samples_a.append(samples_a[-1])
            samples_b.append(b.copy())
            samples_Z.append(samples_Z[-1])
            accept_count += 1
            Hamiltonian_p.append(proposed_H)
            LogL.append(loglikelihood(G,samples_Z[-1],samples_a[-1],b))
        else:
            samples_a.append(samples_a[-1])
            samples_b.append(samples_b[-1])
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
    samples_b = np.array([np.float64(s.item()) if isinstance(s, np.ndarray) else np.float64(s) for s in samples_b[1:]])[warmup*number_of_parameters:-1:number_of_parameters]
    Hamiltonian_p = np.array([np.float64(s.item()) if isinstance(s, np.ndarray) else np.float64(s) for s in Hamiltonian_p[1:]])[warmup*number_of_parameters:-1:number_of_parameters]
    LogL = np.array([np.float64(s.item()) if isinstance(s, np.ndarray) else np.float64(s) for s in LogL[1:]])[warmup*number_of_parameters:-1:number_of_parameters]
    acep_rate_history = np.array(acep_rate_history)[warmup:]

    return samples_Z, samples_a, samples_b, Hamiltonian_p, LogL, acep_rate_history

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