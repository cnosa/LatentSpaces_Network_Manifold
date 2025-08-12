# LATENTSPACES_NETWORK_MANIFOLD

This repository contains experiments and analyses of latent space models applied to network data, using both Euclidean and spherical geometries. The main goal is to assess how well these models capture social structure across various historical and synthetic networks.

## Repository Structure

- **`LATENTSPACES_NETWORK_MANIFOLD/`**
  - `AdditionalContent/` # Extra resources (images, tests)
  - `Example_Karate/` # Karate club example data & notebook
  - `Example_Monks/` # Monks network example & data
  - `FlorentineFamilies/` # Florentine families dataset & analysis
    - `UsingHMC/` # HMC implementation, results & notebooks
  - `LSMN_P.py` # Main Python script
  - `README.md`
  - `requirements.txt`
## Models

We implement latent space models in the following geometries:

- **$\mathbb{R}^1, \mathbb{R}^2, \mathbb{R}^3$**: Euclidean latent spaces
- **$\mathbb{S}^1, \mathbb{S}^2$**: Spherical latent spaces

Each model infers a posterior distribution over node positions, which is then used to interpret structural features of the observed network.

## Case Studies

- **FlorentineFamilies**: Marriage network among 15th-century Florentine families, as analyzed by Padgett & Ansell (1993).
- **Karate Club**: Zachary’s karate club network (1977), used as a benchmark example.
- **Monks**: Placeholder for future experiments with the monk social network dataset.

## Requirements

This project requires the following Python libraries:

- [networkx](https://networkx.org/) — Creation and manipulation of graphs.
- [numpy](https://numpy.org/) — Numerical operations and random data generation.
- [matplotlib](https://matplotlib.org/) — Static data visualization.
- [pandas](https://pandas.pydata.org/) — Data manipulation and analysis using DataFrames.
- [seaborn](https://seaborn.pydata.org/) — Statistical data visualization.
- [scipy](https://scipy.org/) — Scientific functions and tools, including `scipy.stats` and `scipy.special`.
- [plotly](https://plotly.com/python/) — Interactive visualizations (`plotly.express`, `plotly.graph_objects`).
- [tqdm](https://tqdm.github.io/) — Progress bars for iterations.
- [ridgeplot](https://pypi.org/project/ridgeplot/) — Overlaid density plots.
- [matplotlib.colors](https://matplotlib.org/stable/api/colors_api.html) — Color manipulation and conversion.
- [matplotlib.ticker](https://matplotlib.org/stable/api/ticker_api.html) — Control of labels and ticks in plots.

You can install all dependencies with:

```bash
pip install -r requirements.txt