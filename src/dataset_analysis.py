import pickle
import cupy as cy
from matplotlib import pyplot as plt

with open("data/metrics/train_loss.pkl", "rb") as file:
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

plt.show()