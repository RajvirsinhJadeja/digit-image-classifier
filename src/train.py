import cupy as cy
import csv
import time
from mood_neural_network import neuralNetwork
from sqlite_dict import SqliteDict

start_time = time.time()

with open("data/training.csv", "r") as file:
    reader = csv.reader(file)
    train_data = list(reader)

with open("data/validation.csv", "r") as file:
    reader = csv.reader(file)
    val_data = list(reader)

label_to_index = {
    "happy": 0,
    "sad": 1,
    "anger": 2,
    "fear": 3,
    "disgust": 4
}

nn = neuralNetwork()
db = SqliteDict()
train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

batch_size = 32

def get_mean(word_list):
    value_list = []
    for word in word_list:
        value = db[word]
        if value is not None:
            value_list.append(value)
    
    if len(value_list) == 0:
        return cy.zeros(100, dtype=cy.float32)
    
    return cy.mean(cy.array(value_list), axis=0)

def train():
    for start in range(0, len(train_data), batch_size):
        batch = train_data[start : start+batch_size]
        
        input_batch = []
        target_batch = []
        
        for row in batch:
            input_batch.append(get_mean(row[0].lower().split()))
            
            l = cy.zeros(5)
            l[label_to_index[row[1]]] = 1
            target_batch.append(l)
        
        input_batch = cy.array(input_batch)     # shape 32x100
        target_batch = cy.array(target_batch)   # shape 32x5
        
        # Forward Pass
        z_list, activation_list = nn.forward_pass(x=input_batch)
        
        # Loss
        
        break
    return


def validation():
    
    return


def run_epoch(epoch_count):
    for count in range(0, epoch_count):
        train()
        validation()
        break


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
    nn.load_weights_biases()
    
    run_epoch(epoch_count=5)
    # save_data()
