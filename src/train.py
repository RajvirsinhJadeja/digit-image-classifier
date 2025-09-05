import cupy as cy
import pickle
import csv
import time
from mood_neural_network import neuralNetwork

start_time = time.time()

with open("data/training.csv", "r") as file:
    reader = csv.reader(file)
    train_data = list(reader)

with open("data/validation.csv", "r") as file:
    reader = csv.reader(file)
    val_data = list(reader)

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
train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

def run_epoch(epoch_count):
    for count in range(0, epoch_count):
        total_loss = 0
        accuracy_count = 0
        for i in range(0, len(train_data)):
            input = cy.zeros(10000, dtype=cy.float32)
            target = cy.zeros(5, dtype=cy.float32)

            for word in (train_data[i][0]).split():
                idx = map.get(word)
                if idx is not None:
                    input[idx] += 1

            input = input/cy.sqrt(cy.sum(cy.square(input)) + 1e-8)
            
            z_list, activation_list = nn.forward_pass(x=input)
            
            label_index = label_to_index.get(train_data[i][1])
            total_loss += -cy.log(activation_list[-1][label_index] +  1e-9)
            
            if cy.argmax(activation_list[-1]) == label_index:
                accuracy_count += 1
            
            target[label_index] = 1
            nn.backpropogate(z_list=z_list, activation_list=activation_list, target=target, lr=0.001)
        
        train_loss_list.append(total_loss/len(train_data))
        train_accuracy_list.append(accuracy_count/len(train_data))
        
        nn.set_weights_biases(number=count)
        
        total_loss = 0
        accuracy_count = 0
        for i in range(0, len(val_data)):
            input = cy.zeros(10000, dtype=cy.float32)
            target = cy.zeros(5, dtype=cy.float32)

            for word in (val_data[i][0]).split():
                idx = map.get(word)
                if idx is not None:
                    input[idx] += 1

            input = input/cy.sqrt(cy.sum(cy.square(input)) + 1e-8)
            
            z_list, activation_list = nn.forward_pass(x=input)
            
            label_index = label_to_index.get(val_data[i][1])
            total_loss += -cy.log(activation_list[-1][label_index])
            
            if cy.argmax(activation_list[-1]) == label_index:
                accuracy_count += 1
        
        val_loss_list.append(total_loss/len(train_data))
        val_accuracy_list.append(accuracy_count/len(train_data))


def save_data():
    with open("data/metrics/train_loss.pkl", "wb") as file:
            pickle.dump(train_loss_list, file)
    with open("data/metrics/train_accuracy.pkl", "wb") as file:
            pickle.dump(train_accuracy_list, file)
    
    with open("data/metrics/val_loss.pkl", "wb") as file:
            pickle.dump(val_loss_list, file)
    with open("data/metrics/val_accuracy.pkl", "wb") as file:
            pickle.dump(val_accuracy_list, file)


if __name__ ==  "__main__":
    nn = neuralNetwork()
    run_epoch(epoch_count=20)
    save_data()
    
    