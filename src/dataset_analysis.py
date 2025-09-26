import pickle
import csv
import numpy as np
import cupy as cy
from matplotlib import pyplot as plt


def create_npz(name):
    label_to_target = {
        "0":   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "1":   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "2":   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "3":   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "4":   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "5":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "6":   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "7":   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "8":   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "9":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }

    with open(f"data/{name}.csv", "r") as file:
        reader = csv.reader(file)
        train_data = list(reader)

    def get_array(row):
        l = []
        
        for i in range(1, len(row)):
            l.append(int(row[i]))
        
        return np.array(l)
        
    
    embeddings = []
    targets = []

    for row in train_data[1:]:
        emb = get_array(row)
        target = np.array(label_to_target[row[0]], dtype=np.int32)

        embeddings.append(emb)
        targets.append(target)

    embeddings = np.array(embeddings)
    targets = np.array(targets)

    np.savez(f"data/{name}.npz", embeddings=embeddings, targets=targets)


def analyze_dataset(name):
    with open(f"data/{name}.csv", "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0
    num_6 = 0
    num_7 = 0
    num_8 = 0
    num_9 = 0
    
    for row in data:
        if row[0] == "0":
            num_0 += 1
        elif row[0] == "1":
            num_1 += 1
        elif row[0] == "2":
            num_2 += 1
        elif row[0] == "3":
            num_3 += 1
        elif row[0] == "4":
            num_4 += 1
        elif row[0] == "5":
            num_5 += 1
        elif row[0] == "6":
            num_6 += 1
        elif row[0] == "7":
            num_7 += 1
        elif row[0] == "8":
            num_8 += 1
        elif row[0] == "9":
            num_9 += 1

    print("Num 0: ", num_0)
    print("Num 1: ", num_1)
    print("Num 2: ", num_2)
    print("Num 3: ", num_3)
    print("Num 4: ", num_4)
    print("Num 5: ", num_5)
    print("Num 6: ", num_6)
    print("Num 7: ", num_7)
    print("Num 8: ", num_8)
    print("Num 9: ", num_9)
    print("Total: ", num_0+num_1+num_2+num_3+num_4+num_5+num_6+num_7+num_8+num_9)


def showGraph():
    with open("data/metrics/train_loss.pkl", "rb") as file:
        t_loss = pickle.load(file)
        
    with open("data/metrics/train_accuracy.pkl", "rb") as file:
        t_acc = pickle.load(file)
        
    with open("data/metrics/val_loss.pkl", "rb") as file:
        v_loss = pickle.load(file)
        
    with open("data/metrics/val_accuracy.pkl", "rb") as file:
        v_acc = pickle.load(file)


    # Convert to lists (cupy -> python lists)
    t_loss = cy.array(t_loss).tolist()
    t_acc = cy.array(t_acc).tolist()
    v_loss = cy.array(v_loss).tolist()
    v_acc = cy.array(v_acc).tolist()

    # X values = epochs
    epochs = list(range(1, len(t_loss) + 1))

    # --- Plot Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, t_loss, label="Training Loss")
    plt.plot(epochs, v_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, t_acc, label="Training Accuracy")
    plt.plot(epochs, v_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ ==  "__main__":
    showGraph()
    #analyze_dataset("mnist_train")
    #create_npz("mnist_test")