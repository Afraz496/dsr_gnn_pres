# Temporal Graph Neural Networks

This repo houses a lot of code to help you in your tGNN endeavors! 

## Setup

A full setup guide is available [here](https://github.com/Afraz496/dsr_gnn_pres/blob/main/tensorflow_setup.md)

## Usage

The repo is structured as follows

1. `cpox`: Everything related to [Chicken Pox](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting)
2. `engcovid`: Everthing related to [COVID-19 in England](https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/encovid.html). Full paper available [here](https://arxiv.org/pdf/2009.08388)
3. `Presentation`: All scripts and materials used to make the presentation `GNN_DSR.pptx` available in the same folder

### EngCovid

To use the scripts to rerun the analysis and make the dashboard available online first do:

```bash
python encovid_example.py
```

This will train the `MPP LSTM` and `CatBoost` on the torch geometric temporal dataset and save the model and create the materials necessary for the dashboard.

To run the dashboard do:

```bash
python encovid_dashboard.py
```

### Using torch geometric with your own data

I have composed a function which you can adapt to use on your own datasets to make a torch geometric temporal friendly dataset available [here](https://github.com/Afraz496/dsr_gnn_pres/blob/main/use_for_custom_data.py)
