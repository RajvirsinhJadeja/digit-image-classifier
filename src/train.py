import numpy as np
import pickle
import csv
import time
from mood_neural_network import neuralNetwork

start_time = time.time()

with open("data/training.csv", "r") as file:
    reader = csv.reader(file)
    data = list(reader)

with open("data/word_map.pkl", "rb") as file:
    map = pickle.load(file)
    

label_to_index = {
    "happy": 0,
    "sad": 1,
    "anger": 2,
    "fear": 3,
    "disgust": 4
}

nn = neuralNetwork()
total_loss = 0

try:
    for i in range(0, len(data)):
        input = np.zeros(10000)

        for word in (data[0][0]).split():
            idx = map.get(word)
            if idx is not None:
                input[idx] += 1

        if i % 1000 == 0:
            print(i)
        
        z_list, activation_list = nn.forward_pass(x=input)
        total_loss += -np.log(activation_list[-1][label_to_index.get(data[i][1])])

finally:
    elapsed_time = time.time() - start_time
    print("Total Loss: ", total_loss)
    print("Average Loss: ", total_loss/len(data))
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
