import pennylane as qml
from pennylane import numpy as np

dev = qml.device('qiskit.aer' , wires = 2, shots = 1)

@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires = 0)
    qml.RX(params[1], wires = 1)
    qml.CNOT(wires = [0,1])
    return qml.probs(wires = [0,1])

#cost function
def cost(params):
    z = circuit(params)
    s = (z[0]**2) + ((0.5 - z[1])**2) + ((0.5 - z[2])**2) + (z[3]**2)
    return s

#Optimizing the cost function using gradient descent
opt = qml.GradientDescentOptimizer(stepsize = 0.4)
steps = 100
params = np.array([1,1]) 
#running Optimization
for i in range(steps):
    params = opt.step(cost, params)
    if (i+1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))


print("Optimized rotation angles: {}".format(params))