import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x):
    return 1/ (1+ np.exp(-x))

x = np.linspace(-1,1,1000)
plt.plot(x, logistic_function(x))
plt.title("logistic function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
