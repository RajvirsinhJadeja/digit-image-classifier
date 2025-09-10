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