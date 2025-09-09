import pickle
import csv
import cupy as cy
from matplotlib import pyplot as plt
from sqlite_dict import SqliteDict

"""with open("data/metrics/train_loss.pkl", "rb") as file:
    t_loss = pickle.load(file)
    
with open("data/metrics/train_accuracy.pkl", "rb") as file:
    t_acc = pickle.load(file)
    
with open("data/metrics/val_loss.pkl", "rb") as file:
    v_loss = pickle.load(file)
    
with open("data/metrics/val_accuracy.pkl", "rb") as file:
    v_acc = pickle.load(file)


x_value = list(range(1, 6))
y_value = cy.array(t_loss).tolist()

plt.plot(x_value, y_value)

plt.show()

y_value = cy.array(t_acc).tolist()

plt.plot(x_value, y_value)

plt.show()

y_value = cy.array(v_loss).tolist()

plt.plot(x_value, y_value)

plt.show()

y_value = cy.array(v_acc).tolist()

plt.plot(x_value, y_value)

plt.show()"""

db = SqliteDict()

label_to_target = {
    "happy": [1, 0, 0, 0, 0],
    "sad": [0, 1, 0, 0, 0],
    "anger": [0, 0, 1, 0, 0],
    "fear": [0, 0, 0, 1, 0],
    "disgust": [0, 0, 0, 0, 1]
}

new_data = []

def get_mean(word_list):
    value_list = []
    for word in word_list:
        value = db[word]
        if value is not None:
            value_list.append(value)
    
    if len(value_list) == 0:
        return cy.zeros(100, dtype=cy.float32)
    
    return cy.mean(cy.array(value_list), axis=0)

with open("data/test.csv", "r") as file:
    reader = csv.reader(file)
    train_data = list(reader)

for row in train_data:
    emb = get_mean(row[0].lower().split())
    target = label_to_target[row[1]]
    
    emb_string = " ".join(map(str, emb.tolist()))
    target_string = " ".join(str(item) for item in target)
    
    new_data.append([emb_string, target_string])
    
    # convert back to floatsfloat_list = list(map(float, test.split()))
    
with open("data/precomputed_test.csv", "w",  newline="") as file:
    writer = csv.writer(file)
    writer.writerows(new_data)
    