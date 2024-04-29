import numpy as np
import matplotlib.pyplot as plt

def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))


x = np.linspace(-10, 10, 400)

y = mish(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Mish Activation Function')
plt.title('Mish Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
