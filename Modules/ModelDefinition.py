import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import pandas as pd
from tqdm import tqdm
np.random.seed(42)


def loglikelihood(Z,a):
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

def grad_loglikelihood(Z,a):
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