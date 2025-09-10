import cupy as cy
import pickle
import csv
import time
from mood_neural_network import neuralNetwork

start_time = time.time()

with open("data/precomputed_training_test.csv", "r") as file:
    reader = csv.reader(file)
    train_data = list(reader)

with open("data/precomputed_validation.csv", "r") as file:
    reader = csv.reader(file)
    val_data = list(reader)

nn = neuralNetwork()
train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

batch_size = 32

def train():
    total_loss = 0
    total_acc = 0
    for start in range(0, len(train_data), batch_size):
        batch = train_data[start : start+batch_size]
        
        input_batch = [[float(num) for num in row[0].split()] for row in batch]
        target_batch = [[int(num) for num in row[1].split()] for row in batch]
        
        input_batch = cy.array(input_batch)     # shape 32x100
        target_batch = cy.array(target_batch)   # shape 32x5
        
        # Forward Pass
        z_list, activation_list = nn.forward_pass(x=input_batch)
        
        # Loss/Acc
        probability_list = activation_list[-1]  # shape 32x5
        
        batch_loss = -cy.sum(target_batch * cy.log(probability_list + 1e-9), axis=1)
        total_loss += cy.sum(batch_loss)
        
        batch_acc = cy.sum(cy.argmax(probability_list, axis=1) == cy.argmax(target_batch, axis=1))
        total_acc += batch_acc
        
        #Back propogation
        nn.backpropogate(z_list=z_list, activation_list=activation_list, target=target_batch, batch_size=batch_size)
    
    train_loss_list.append(total_loss/len(train_data))
    train_accuracy_list.append(total_acc/len(train_data))


def validation():
    total_loss = 0
    total_acc = 0
    for start in range(0, len(val_data), batch_size):
        batch = val_data[start : start+batch_size]
        
        input_batch = [[float(num) for num in row[0].split()] for row in batch]
        target_batch = [[int(num) for num in row[1].split()] for row in batch]
        
        input_batch = cy.array(input_batch)     # shape 32x100
        target_batch = cy.array(target_batch)   # shape 32x5
        
        # Forward Pass
        z_list, activation_list = nn.forward_pass(x=input_batch)
        
        # Loss/Acc
        probability_list = activation_list[-1]  # shape 32x5
        
        batch_loss = -cy.sum(target_batch * cy.log(probability_list), axis=1)
        total_loss += cy.sum(batch_loss)
        
        batch_acc = cy.sum(cy.argmax(probability_list, axis=1) == cy.argmax(target_batch, axis=1))
        total_acc += batch_acc
    
    val_loss_list.append(total_loss/len(val_data))
    val_accuracy_list.append(total_acc/len(val_data))


def run_epoch(epoch_count):
    for count in range(0, epoch_count):
        print("\nTraining #", count)
        train()
        print("Validation #", count)
        validation()


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
    
    run_epoch(epoch_count=1000)
    print("Time: ", time.time() - start_time)
    save_data()
