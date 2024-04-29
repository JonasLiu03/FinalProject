import numpy as np
import matplotlib.pyplot as plt

def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)


x_values = np.linspace(-2, 2, 400)
alpha = 0.25
y_values = prelu(x_values, alpha)
plt.figure(figsize=(8, 4))
plt.plot(x_values, y_values, label=f'PReLU, α = {alpha}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('PReLU Activation Function')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()
