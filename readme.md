# Handwritten Digit Recognition (NumPy from scratch)

A neural network built with only NumPy and pandas (no TensorFlow or PyTorch) that recognizes handwritten digits from the MNIST dataset.

- Architecture: 784 inputs → 10 hidden neurons (ReLU) → 10 outputs (softmax)
- Training: gradient descent with backpropagation

## Files
- `handdigit.py`: training code
- `test.py`: testing code
- `weights.npz`: trained weights

## Setup
pip install numpy pandas matplotlib

## Data
Download `train.csv` from the Kaggle "Digit Recognizer" competition and place it in this folder.

## Run
python handdigit.py