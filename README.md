# MNIST Classifier

This is a simple MNIST classifier using a simple neural network with CNN and fully connected layers.

The model is trained using the MNIST dataset and is written in PyTorch.

The model is deployed using Streamlit.

## Online Deployment

You can use the online classsifier at [mnist-classifier](https://littlefish-coder-mnist-classifier.streamlit.app/).

## Environment Setup

1. create a virtual environment via conda or venv.
2. install the required packages using the `requirements.txt` file.

```bash
conda create -n mnist-classifier python=3.10
```

```bash
conda activate mnist-classifier
```

```bash
pip install -r requirements.txt
```

## Run UI

You can test the model using the streamlit app.

```bash
streamlit run app.py
```
