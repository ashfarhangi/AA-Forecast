AA-Forecast: Anomaly-Aware Forecast for Extreme Events

[![Conference](http://img.shields.io/badge/ECML-2022-4b44ce.svg)](https://arxiv.org/abs/2208.09933)
</div>
==============================

If you find this code or idea useful, please consider citing our work:

```@article{farhangi2022aa,
  title={AA-Forecast: Anomaly-Aware Forecast for Extreme Events},
  author={Farhangi, Ashkan and Bian, Jiang and Huang, Arthur and Xiong, Haoyi and Wang, Jun and Guo, Zhishan},
  journal={arXiv preprint arXiv:2208.09933},
  year={2022}
}
```
## Getting Started

Instructions on setting up your project locally or on a cloud platform. To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.

- Tensorflow 2.1.1
- Nvidia GPU 
### Datasets
Datasets are located in the data folder:

credit-card-sales-covid-19.csv
electricity.csv
tax-sales-hurricane.csv

### Installation

1. Clone the repo.

   ```
   git clone https://github.com/0415070/AA-RNN.git
   ```

2. Install requirement packages.

   ```
   pip install -r requirements.txt
   ```

3. Run model.py after the dataset has been gathered.
You can use  make_data.py for this.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.







![Figure 1-1](https://raw.githubusercontent.com/ashfarhangi/AA-Forecast/main/visualization/Decomposition.png "Figure 1-1")




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


