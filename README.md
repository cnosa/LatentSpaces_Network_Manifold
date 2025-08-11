# LATENTSPACES_NETWORK_MANIFOLD

This repository contains experiments and analyses of latent space models applied to network data, using both Euclidean and spherical geometries. The main goal is to assess how well these models capture social structure—such as centrality and clustering—across various historical and synthetic networks.

## Repository Structure



## Models

We implement latent space models in the following geometries:

- **\(\mathbb{R}^1\), \(\mathbb{R}^2\), \(\mathbb{R}^3\)**: Euclidean latent spaces
- **\(\mathbb{S}^1\), \(\mathbb{S}^2\)**: Spherical latent spaces

Each model infers a posterior distribution over node positions, which is then used to interpret structural features of the observed network.

## Case Studies

- **FlorentineFamilies**: Marriage network among 15th-century Florentine families, as analyzed by Padgett & Ansell (1993).
- **Karate Club**: Zachary’s karate club network (1977), used as a benchmark example.
- **Monks**: Placeholder for future experiments with the monk social network dataset.

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `pickle` (for result serialization)
