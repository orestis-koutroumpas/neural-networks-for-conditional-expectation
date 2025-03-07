# Conditional Expectation Estimation with Neural Networks

This repository implements the estimation of conditional expectations using neural networks. Specifically, the neural network models are trained to approximate the conditional expectations 𝔼𝑦[𝐺(𝒴)|𝒳=𝑋] for different functions G(Y), using various function families such as [A1], [A2], and [C1]. The code also provides numerical solutions for comparison.

## Function Families
### [Α]: Range \( −∞,∞ \)

- **[A1]**: 
  - 𝜔(𝓏) = 𝓏, 
  - 𝜌(𝓏) = −1,
  - 𝜑(𝓏) = z^2 / 2,
  - 𝜓(𝓏) = −𝓏 

- **[A2]**: 
  - 𝜔(𝓏) = sinh(𝓏),
  - 𝜌(𝓏) = −𝑒^(−0.5|𝓏|),
  - 𝜑(𝓏) = 𝑒^(0.5|𝓏|−1) + (1/3)(𝑒^(−1.5|𝓏|−1)),
  - 𝜓(𝓏) = 2 𝑠𝑖𝑔𝑛(𝓏) (𝑒^(−0.5|𝓏|−1))

### [C]: Range (a, b)

- **[C1]**:
  - 𝜔(𝓏) = 𝑎 / (1+𝑒^𝓏) + 𝑏 / (1 + 𝑒^𝓏,
  - 𝜌(𝓏) = − 𝑒^𝓏 / (1+𝑒^𝓏),
  - 𝜑(𝓏) = (𝑏−𝑎) / (1 + 𝑒^𝓏)+ 𝑏 log(1+ 𝑒^𝓏),
  - 𝜓(𝓏) = −log(1+𝑒^𝓏)

## Overview



## Neural Network Architecture

- The network consists of an input layer, a hidden layer, and an output layer.
- The forward pass uses the ReLU activation function.
- The backward pass computes gradients and updates the weights using the Adam optimizer.

## Training

The neural networks are trained using the following parameters:
- **Learning Rate**: 0.001
- **Hidden Size**: 50
- **Epochs**: 2000
- **Optimizer**: Adam with default parameters.

The training loop computes the loss at each epoch and updates the weights accordingly. After training, the network’s performance is visualized.

## Results

The repository contains plots that show the comparison between the neural network predictions and the numerical solutions for various functions G(Y). The learning curves and results for different functions are plotted to demonstrate the neural network’s ability to approximate conditional expectations.

### Learning Curves and Approximations

#### For G(Y) = Y

1. Learning Curves with [A1] and [A2]:

   <div style="display: flex; justify-content: space-between; text-align: center; width: 100%;">
      <img src="screenshots/curve1.png" alt="Learning Curve A1" width="45%" />
      <img src="screenshots/curve2.png" alt="Learning Curve A2" width="45%" />
   </div>

2. Approximation

   <div style="text-align: center;">
      <img src="screenshots/curve3.png" alt="Approximation for G(Y) = Y" width="90%" />
   </div>

#### For G(Y) = min{1, max{-1,Y}}

1. Learning Curve with [A1] and [C1]:

   <div style="display: flex; justify-content: space-between; text-align: center; width: 100%;">
      <img src="screenshots/curve4.png" alt="Learning Curve A1" width="45%" />
      <img src="screenshots/curve5.png" alt="Learning Curve C1" width="45%" />
   </div>

2. Approximation:
   <div style="text-align: center;">
      <img src="screenshots/curve6.png" alt="Approximation for G(Y) = min{1, max{-1,Y}}" width="90%" />
   </div>


## Usage

To run the code:

1. Clone the repository:

       git clone https://github.com/orestis-koutroumpas/neural-networks-for-conditional-expectation.git

2. Navigate into the repository directory:

        cd folder-name

3.  Install required dependencies:

        pip install numpy matplotlib scipy

4. Run the main script:

        python main.py

This will train the neural network models and generate the plots.

