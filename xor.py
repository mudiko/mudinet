from layer import Linear, Tanh
import numpy as np
from nn import NeuralNet
from train import train

inputs=np.array([[0,0],[1,0],[0,1],[1,1]])
targets=np.array([[0],[1],[1],[0]])

net=NeuralNet([
    Linear(input_size=2,output_size=2),
    Tanh(),
    Linear(input_size=2,output_size=2),
    Tanh(),
    Linear(input_size=2,output_size=2),
    Tanh(),
    Linear(input_size=2,output_size=1)
])

train(net, inputs, targets, num_epochs=2000)

for x,y in zip(inputs,targets):
    predicted=net.forward(x)
    print(x, predicted)
