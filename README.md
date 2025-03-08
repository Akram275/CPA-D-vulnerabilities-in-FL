# Federated Learning with RLWE-like Perturbations

## Overview
This repository contains implementations for training neural networks with RLWE-like perturbations in both standalone and Federated Learning (FL) settings. The primary focus is on:
- Injecting RLWE-like noise into model parameters during training.
- Simulating a Federated Learning setup where a perturbation occurs at a specific iteration.

## Repository Structure

- **`DenseNet_mnist.py`**: 
  - Trains a DenseNet classifier on the MNIST dataset.
  - Replaces a portion of the model parameters with RLWE-like noise during training.
  
- **`ResNET_mnist.py`**: 
  - Trains a ResNet classifier on the MNIST dataset.
  - Does not include perturbation but serves as a baseline for comparison.
  
- **`FL_with_CPAD_perturbation.py`**: 
  - Simulates a Federated Learning (FedAvg-based) training scenario.
  - At a specific iteration, the server replaces part of the global model parameters with RLWE-like noise.
  
- **`krum_sgd_mnist.py`**: 
  - Implements the Krum aggregation rule in an FL setting to mitigate the impact of adversarial clients.
  - Evaluates the effect of aggregation on model robustness.

## Installation
Ensure you have the required dependencies installed:
```bash
pip install numpy tensorflow pandas
```

## Usage

### 1. Training a DenseNet with RLWE-like Perturbations
```bash
python DenseNet_mnist.py
```

### 2. Training a ResNet Baseline
```bash
python ResNET_mnist.py
```

### 3. Running Federated Learning with Perturbations
```bash
python FL_with_CPAD_perturbation.py
```

### 4. Running Federated Learning with Krum Aggregation
```bash
python krum_sgd_mnist.py
```


## License
This project is licensed under the MIT License.

