import pickle
import cupy as cy

with open("data/weights_biases.pkl", "rb") as file:
    data = pickle.load(file)

with open("data/saved_models/weights_biases_epoch19.pkl", "rb") as file:
    new_data = pickle.load(file)

with open("data/metrics/val_loss.pkl", "rb") as file:
    loss = pickle.load(file)
    

old_bias = cy.array(data["bias5"])
new_bias = cy.array(new_data["bias5"])

print(old_bias)
print(new_bias)

print(cy.array(loss))