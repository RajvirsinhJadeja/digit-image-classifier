import cupy as cy
import pickle
import time
from mood_neural_network import neuralNetwork
from dataset_analysis import showGraph

start_time = time.time()

train_data = cy.load("data/mnist_train.npz")
train_embeddings = cy.array(train_data["embeddings"])
train_targets = cy.array(train_data["targets"])

val_data = cy.load("data/mnist_test.npz")
val_embeddings = cy.array(val_data["embeddings"])
val_targets = cy.array(val_data["targets"])

nn = neuralNetwork()
train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

batch_size = 64


def train():
    total_loss = 0
    total_acc = 0
    for start in range(0, len(train_embeddings), batch_size):
        input_batch = train_embeddings[start:start+batch_size]  # shape 32xinput
        target_batch = train_targets[start:start+batch_size]    # shape 32x10
        
        # Forward Pass
        z_list, activation_list = nn.forward_pass(x=input_batch, dropout_rate=0)
        
        # Loss/Acc
        probability_list = activation_list[-1]  # shape 32x10
        
        batch_loss = -cy.sum(target_batch * cy.log(probability_list + 1e-9), axis=1)
        total_loss += cy.sum(batch_loss)
        
        batch_acc = cy.sum(cy.argmax(probability_list, axis=1) == cy.argmax(target_batch, axis=1))
        total_acc += batch_acc
        
        #Back propogation
        gradient_weights, gradient_biases = nn.backpropagate(z_list=z_list, activation_list=activation_list, target=target_batch, batch_size=batch_size)
        nn.adam_optimizer(gradient_weights=gradient_weights, gradient_biases=gradient_biases)
    
    train_loss_list.append(total_loss/len(train_embeddings))
    train_accuracy_list.append(total_acc/len(train_embeddings))


def validation():
    total_loss = 0
    total_acc = 0
    for start in range(0, len(val_embeddings), batch_size):
        input_batch = val_embeddings[start:start+batch_size]  # shape 32xinput
        target_batch = val_targets[start:start+batch_size]    # shape 32x5
        
        # Forward Pass
        z_list, activation_list = nn.forward_pass(x=input_batch, dropout_rate=0)
        
        # Loss/Acc
        probability_list = activation_list[-1]  # shape 32x5
        
        batch_loss = -cy.sum(target_batch * cy.log(probability_list), axis=1)
        total_loss += cy.sum(batch_loss)
        
        batch_acc = cy.sum(cy.argmax(probability_list, axis=1) == cy.argmax(target_batch, axis=1))
        total_acc += batch_acc
    
    val_loss_list.append(total_loss/len(val_embeddings))
    val_accuracy_list.append(total_acc/len(val_embeddings))


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
    
    run_epoch(epoch_count=100)
    print("Time: ", time.time() - start_time)
    save_data()
    nn.save_weights_biases()
    showGraph()