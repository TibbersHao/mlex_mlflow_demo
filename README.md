# MLflow Demo

![GitHub All Releases](https://img.shields.io/github/downloads/TibbersHao/mlflow_demo/total.svg)
![GitHub stars](https://img.shields.io/github/stars/TibbersHao/mlflow_demo.svg?style=social)
![GitHub forks](https://img.shields.io/github/forks/TibbersHao/mlflow_demo.svg?style=social)

This repository demonstrates several MLFlow capabilities regarding experiment tracking to model registry.

Two demonstrations have been included here.

## MNIST Recognizer

This demonstration is adapted from the original [MLFlow example](https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST) that demonstrates autolog with pre-defined configuration files.

In order to run, please refer to the README.md in the `mnist_recognizer` sub-directory.

## Pytorch Autoencoder using MNIST Dataset

This demonstration includes setting up a local MLFlow tracking server to log a pytorch lightning autoencoder with some hyperparameter tuning capabilities. 

In order to run:

1. Create a conda environment and install dependencies using `requirements.txt`.

```
pip install -r requirements.txt
```

2. In terminal, setup the tracking server using make file command.

```
make init_server
```

3. Navigate to the `pytorch_autoencoder` directory

for autolog example, run:
```
python autolog_example.py
```

for manual log example with model schema demonstration, run:
```
python manual_log_example.py
```

4. For navigation, open a web browser and naviage to the uri provided during server setup. The default port for this example has been set to `8080` and can be changed in the `.env` file. One example for the corresponding uri would be: `localhost:8080`




