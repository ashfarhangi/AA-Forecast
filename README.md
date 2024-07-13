# AA-Forecast: Anomaly-Aware Forecast for Extreme Events

**Published in**: Data Mining and Knowledge Discovery, 2023


## Overview

Time series models are often impacted by extreme events and anomalies, which are prevalent in real-world datasets. Such models require careful probabilistic forecasts, vital in risk management for extreme events like hurricanes and pandemics. Our proposed framework, AA-Forecast, leverages the effects of anomalies to improve prediction accuracy during extreme events. 

## Key Features

- **Automatic Anomaly Detection**: The model extracts anomalies automatically and incorporates them through an attention mechanism to enhance forecast accuracy during extreme events.
- **Dynamic Uncertainty Optimization**: Employs an algorithm that reduces the uncertainty of forecasts in a nonlinear manner.
- **Superior Performance**: Demonstrates consistent superior accuracy with less uncertainty across different datasets compared to current prediction models.

## Contributions

1. **Anomaly Extraction**: Introduces a novel decomposition method to extract anomalies.
2. **Attention Mechanism**: Leverages the extracted anomalies through a specialized attention mechanism to improve forecasting.
3. **Dynamic Optimization**: Reduces uncertainty in forecasts with a dynamic optimization approach.

## Datasets
The framework is evaluated on the following datasets:

Hurricane Data: Time series data related to hurricane events.
Pandemic Data: COVID-19 impact on various sectors.
Synthetic Data: Generated datasets to test the model under controlled conditions.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ashfarhangi/AA-Forecast&type=Date)](https://star-history.com/#ashfarhangi/AA-Forecast&Date)


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




![Figure 1-1](https://raw.githubusercontent.com/ashfarhangi/AA-Forecast/main/visualization/Decomposition.png "Figure 1-1")


[![Conference](http://img.shields.io/badge/ECML-2022-4b44ce.svg)](https://arxiv.org/abs/2208.09933)
</div>


The citation for the paper, code, and data is as below:

```bibtex
@article{farhangi2022aa,
  title={AA-Forecast: Anomaly-Aware Forecast for Extreme Events},
  author={Farhangi, Ashkan and Bian, Jiang and Huang, Arthur and Xiong, Haoyi and Wang, Jun and Guo, Zhishan},
  journal={Data Mining and Knowledge Discovery},
  year={2023},
  volume={37},
  pages={1209-1229},
  doi={10.1007/s10618-023-00919-7}
}
```
