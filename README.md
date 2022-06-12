RAA
==============================

Implementation of Relational Archetypal Analysis (RAA).
RAA is built upon the Archetypal Analysis (AA) method proposed by Cutler and Breiman. AA is a unsupervised clustering algorithm that learns the extremes of the data, called archetypes. From these archetypes all data points can be expressed as convex combinations. Our contribution is the extension of AA to graph data. 

This implementation makes use of sampling from the sparse representation of the adjacency matrix to improve training time. Addtionally have this implementation been developed for unipartite undirected graphs as well as bipartite graphs. However, it is assumed that the graph consist of a single giant connected component.  

### Prerequisites

To run this project make sure you have the following packages.
* npm
  ```sh
  pip install requirements.txt
  ```

  The implementation uses the [Pytorch sparse](https://github.com/rusty1s/pytorch_sparse) package. Installation details can be found on [Pytorch geometric's webpage](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

  The implementation have both CPU and CUDA capabilities. 


Latent embedding space             |  Archetypal maximum membership ordering of adjacency matrix
:-------------------------:|:-------------------------:
![](/reports/figures/show_embedding_facebook_k3.png)  |  ![](/reports/figures/ordered_adjacency_facebook_k3.pdf)



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
