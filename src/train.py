import numpy as np
import pickle
import csv
from mood_neural_network import neuralNetwork

with open("data/training.csv", "r") as file:
    reader = csv.reader(file)
    data = list(reader)

with open("data/word_map.pkl", "rb") as file:
    map = pickle.load(file)
    

input = [0] * 10000

for word in (data[0][0]).split():
    if map.get(word) is not None:
        input[map.get(word)] += 1

nn = neuralNetwork()
z_list, activation_list = nn.forward_pass(x=input)

print(activation_list[-1])
print("Loss: ", -np.log(activation_list[-1][1]))
print("Loss: ", -np.log(0.85))
