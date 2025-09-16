import pickle
import csv
import cupy as cy
import sqlite3
from sqlite_dict import SqliteDict
from matplotlib import pyplot as plt

def create_10k_npz():
    label_to_target = {
        "happy":   [1, 0, 0, 0, 0],
        "sad":     [0, 1, 0, 0, 0],
        "anger":   [0, 0, 1, 0, 0],
        "fear":    [0, 0, 0, 1, 0],
        "disgust": [0, 0, 0, 0, 1]
    }

    with open("data/training.csv", "r") as file:
        reader = csv.reader(file)
        train_data = list(reader)

    with open("data/word_map.pkl", "rb") as file:
        word_map = pickle.load(file)

    def get_array(input):
        bow = cy.zeros(10000, dtype=cy.int32)
        for word in input:
            x = word_map.get(word)
            if x is not None:
                bow[x] += 1
        return bow

    embeddings = []
    targets = []

    for row in train_data:
        emb = get_array(row[0].lower().split())
        target = cy.array(label_to_target[row[1]], dtype=cy.int32)

        embeddings.append(emb)
        targets.append(target)

    embeddings = cy.array(embeddings)
    targets = cy.array(targets)

    cy.savez("data/precomputed_test.npz", embeddings=embeddings, targets=targets)

def create_300d_db():
    conn = sqlite3.connect("data/word_map.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS kv")
    cursor.execute("""
        CREATE TABLE kv (
            key TEXT PRIMARY KEY,
            value BLOB
        )
    """)

    with open("data/glove.6B.300d.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = cy.array(parts[1:], dtype=cy.float32)

            vector_cpu = vector.get()  
            value_blob = pickle.dumps(vector_cpu, protocol=pickle.HIGHEST_PROTOCOL)

            cursor.execute("INSERT INTO kv (key, value) VALUES (?, ?)", (word, value_blob))
        
        conn.commit()
        conn.close()
        print("Database created: data/word_map.db")

def create_300d_npz():
    label_to_target = {
        "happy":   [1, 0, 0, 0, 0],
        "sad":     [0, 1, 0, 0, 0],
        "anger":   [0, 0, 1, 0, 0],
        "fear":    [0, 0, 0, 1, 0],
        "disgust": [0, 0, 0, 0, 1]
    }

    sd = SqliteDict()
    
    with open("data/validation.csv", "r") as file:
        reader = csv.reader(file)
        train_data = list(reader)

    def get_array(tokens):
        bow = []
        for word in tokens:
            x = sd[word]
            if x is not None:
                bow.append(x)
                
        if len(bow) == 0:
            return cy.zeros(300, dtype=cy.float32)
        
        return cy.mean(cy.array(bow), axis=0)


    embeddings = []
    targets = []

    for row in train_data:
        emb = get_array(row[0].lower().split())
        target = cy.array(label_to_target[row[1]], dtype=cy.int32)

        embeddings.append(emb)
        targets.append(target)

    embeddings = cy.stack(embeddings)
    targets = cy.stack(targets)
    
    print(embeddings.shape)

    cy.savez("data/precomputed_300d_validation.npz", embeddings=embeddings, targets=targets)
    
    print("Created .npz file")

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
    
showGraph()