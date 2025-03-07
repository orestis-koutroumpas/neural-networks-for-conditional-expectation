import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# cdf of h(Y|X)
def H(y, x): return norm.cdf(y - 0.8 * x, loc=0, scale=1)

# G(Y) = Y
def g1(y): return y

# G(Y) = min{1, max{-1,Y}}
def g2(y): return np.minimum(1, np.maximum(-1, y))

# ReLU activation function
def relu(x): return np.maximum(0, x)

# ReLU Derivative
def relu_derivative(x): return np.where(x > 0, 1, 0)

### [A1]
# ω(z)
def omega_A1(z): return z
def rho_A1(z): return -1
def phi_A1(z): return z**2 / 2
def psi_A1(z): return -z

### [A2]
def omega_A2(z): return np.sinh(z)
def rho_A2(z): return -np.exp(-0.5 * np.abs(z))
def phi_A2(z): return np.exp(0.5 * np.abs(z)) - 1 + 0.33333 * (np.exp(-1.5 * np.abs(z)) - 1)
def psi_A2(z): return 2 * np.sign(z) * (np.exp(-0.5 * np.abs(z)) - 1)

### [C1]
def omega_C1(z, a=-1, b=1): return a / (1 + np.exp(z)) + b * np.exp(z) / (1 + np.exp(z))
def rho_C1(z): return - np.exp(z) / (1 + np.exp(z))
def phi_C1(z, a=-1, b=1): return (b-a) / (1 + np.exp(z)) + b * np.log(1 + np.exp(z))
def psi_C1(z): return -np.log(1 + np.exp(z))

# Cost function
def J(u, Y, d_func, phi_func, psi_func):
    return np.mean(phi_func(u) + d_func(Y) * psi_func(u))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.A1 = np.random.normal(0, np.sqrt(1 / input_size), (hidden_size, input_size))
        self.B1 = np.zeros((hidden_size, 1))
        self.A2 = np.random.normal(0, np.sqrt(1 / hidden_size), (output_size, hidden_size))
        self.B2 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

        # Initialize ADAM parameters
        self.m_A1 = np.zeros_like(self.A1)
        self.v_A1 = np.zeros_like(self.A1)
        self.m_B1 = np.zeros_like(self.B1)
        self.v_B1 = np.zeros_like(self.B1)

        self.m_A2 = np.zeros_like(self.A2)
        self.v_A2 = np.zeros_like(self.A2)
        self.m_B2 = np.zeros_like(self.B2)
        self.v_B2 = np.zeros_like(self.B2)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step for ADAM

    def forward(self, X):
        self.W1 = np.dot(self.A1, X) + self.B1
        self.Z1 = relu(self.W1)
        self.W2 = np.dot(self.A2, self.Z1) + self.B2
        self.Y = self.W2
        return self.Y

    def backward(self, X, Y, d_func, omega_func, rho_func): 
        # Compute forward pass
        u = self.forward(X)

        # Compute gradient components
        d_loss = d_func(Y) - omega_func(u)
        v2 = d_loss * rho_func(u)

        # Compute gradients for output layer
        d_A2 = np.dot(v2, self.Z1.T) / X.shape[1]
        d_B2 = np.mean(v2, axis=1, keepdims=True)

        # Compute gradients for hidden layer
        v1 = np.dot(self.A2.T, v2) * relu_derivative(np.dot(self.A1, X) + self.B1)
        d_A1 = np.dot(v1, X.T) / X.shape[1]
        d_B1 = np.mean(v1, axis=1, keepdims=True)

        # Update time step
        self.t += 1
        
        # Update weights and biases using ADAM
        for param, grad, m, v in [
            (self.A1, d_A1, self.m_A1, self.v_A1),
            (self.B1, d_B1, self.m_B1, self.v_B1),
            (self.A2, d_A2, self.m_A2, self.v_A2),
            (self.B2, d_B2, self.m_B2, self.v_B2)
        ]:
            # Update biased first moment estimate
            m[:] = self.beta1 * m + (1 - self.beta1) * grad  
            
            # Update biased second moment estimate
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)  

            # Correct bias for moments
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameters using Adam formula
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X, Y, epochs, d_func, omega_func, rho_func, phi_func, psi_func, g_name, functions_name):
        losses = []
        print(f"\nTraining Neural Network for {g_name} with {functions_name}.")
        print("=" * 50)

        for epoch in range(epochs):
            self.backward(X, Y, d_func, omega_func, rho_func)
            loss = J(self.forward(X), Y, d_func, phi_func, psi_func)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # Plot learning curve
        plt.plot(losses)
        plt.xlabel("Number of Iterations")
        plt.title(f"Learning Curve for {g_name} with {functions_name}")
        plt.show()


def plot_results(X_grid, V, predictions, labels, colors, title):
    plt.figure(figsize=(10, 6))
    plt.plot(X_grid, V, label="Numerical", color="black")
    for pred, label, color in zip(predictions, labels, colors):
        plt.plot(X_grid, pred, label=label, color=color)
    plt.xlim(X_grid.min(), X_grid.max())
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel(r'$\mathbb{E}_y[G(\mathcal{Y}) \mid \mathcal{X} = X]$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    N = 500
    X = np.random.normal(0, 1, (1, N))
    W = np.random.normal(0, 1, (1, N))
    Y = 0.8 * X + W

    ### Numerical solution
    M = 100
    N_samples = 100
    X_grid = np.linspace(X.min(), X.max(), M)
    Y_N = np.linspace(Y.min(), Y.max(), N_samples)

    F = np.zeros((M, N_samples))
    for j in range(M):
        F[j][0] = 0.5 * (H(Y_N[1], X_grid[j]) - H(Y_N[0], X_grid[j]))
        F[j][-1] = 0.5 * (H(Y_N[-1], X_grid[j]) - H(Y_N[-2], X_grid[j]))
        for i in range(2, N_samples):
            F[j][i - 1] = 0.5 * (H(Y_N[i], X_grid[j]) - H(Y_N[i - 2], X_grid[j]))

    G1 = np.array([g1(y) for y in Y_N])
    V1 = np.dot(F, G1)

    G2 = np.array([g2(y) for y in Y_N])
    V2 = np.dot(F, G2)

    ### Neural Network Solution

    # Network parameters
    input_size = 1
    hidden_size = 50
    output_size = 1
    learning_rate = 0.001
    epochs = 2000

    # Create and train the neural networks for G(Y) = Y
    nn_G1_A1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_G1_A1.train(X, Y, epochs, d_func=g1, omega_func=omega_A1, rho_func=rho_A1, phi_func=phi_A1, psi_func=psi_A1, g_name="G(Y) = Y", functions_name="[A1]")

    nn_G1_A2 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_G1_A2.train(X, Y, epochs, d_func=g1, omega_func=omega_A2, rho_func=rho_A2, phi_func=phi_A2, psi_func=psi_A2, g_name="G(Y) = Y", functions_name="[A2]")  
  
    # Predictions
    G1_A1_pred = np.array(omega_A1([nn_G1_A1.forward(np.array([[x]])) for x in X_grid]))
    G1_A2_pred = np.array(omega_A2([nn_G1_A2.forward(np.array([[x]])) for x in X_grid]))

    # Plot results
    plot_results(X_grid, V1, [G1_A1_pred.flatten(), G1_A2_pred.flatten()], ["[A1]", "[A2]"], ["blue", "red"], "Conditional Expectation for G(Y) = Y")

    # Create and train the neural networks for G(Y) = min{1, max{-1,Y}}
    nn_G2_A1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_G2_A1.train(X, Y, epochs, d_func=g2, omega_func=omega_A1, rho_func=rho_A1, phi_func=phi_A1, psi_func=psi_A1, g_name="G(Y) = min[1, max[-1,Y]]", functions_name="[A1]")    

    nn_G2_C1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_G2_C1.train(X, Y, epochs, d_func=g2, omega_func=omega_C1, rho_func=rho_C1, phi_func=phi_C1, psi_func=psi_C1, g_name="G(Y) = min[1, max[-1,Y]]", functions_name="[C1]") 
    
    # Predictions
    G2_A1_pred = np.array(omega_A1([nn_G2_A1.forward(np.array([[x]])) for x in X_grid]))
    G2_C1_pred = np.array(omega_C1([nn_G2_C1.forward(np.array([[x]])) for x in X_grid]))

    # Plot results
    plot_results(X_grid, V2, [G2_A1_pred.flatten(), G2_C1_pred.flatten()], ["[A1]", "[C1]"], ["blue", "red"], "Conditional Expectation for G(Y) = min[1, max[-1, Y]]")
