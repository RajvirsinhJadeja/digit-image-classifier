import numpy as np
import random
import pickle


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator)
    return numerator/denominator


class neuralNetwork:
    def __init__(self):
        with open("data/weights_biases.pkl", "rb") as file:
            data = pickle.load(file)
        
        self.weights = [np.array(data["weight1"]), np.array(data["weight2"]), np.array(data["weight3"]), np.array(data["weight4"]), np.array(data["weight5"])]
        self.biases = [np.array(data["bias1"]), np.array(data["bias2"]), np.array(data["bias3"]), np.array(data["bias4"]), np.array(data["bias5"])]
    
    
    def forward_pass(self, x) -> tuple:
        z_list = []
        activation_list = [x]
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activation_list[i]) + self.biases[i]
            a = relu(z)
            
            z_list.append(z)
            activation_list.append(a)

        z5 = np.dot(self.weights[-1], activation_list[-1]) + self.biases[-1]
        a5 = softmax(z5)
        
        z_list.append(z5)
        activation_list.append(a5)

        return z_list, activation_list
    
    
    def backpropogate(self, z_list, activation_list, target, lr=0.1):
        updated_weights = []
        updated_biases = []
        
        dz_list = [activation_list[-1] - target]
        
        updated_weights.append(self.weights[-1] - (lr * (np.outer(dz_list[-1], activation_list[-2]))))
        updated_biases.append(self.biases[-1] - (lr * dz_list[-1]))
        
        for i in range(4, 0, -1):
            dz = np.dot(self.weights[i].T, dz_list[-1]) * relu_derivative(z_list[i-1])
            
            updated_weights.insert(0, self.weights[i-1] - (lr * (np.outer(dz, activation_list[i-1]))))
            updated_biases.insert(0, self.biases[i-1] - (lr * dz))
            
            dz_list.append(dz)
        
        self.weights = updated_weights
        self.biases = updated_biases


    def get_weights_biases(self) -> tuple:
        return self.weights, self.biases


if __name__ ==  "__main__":
    nn = neuralNetwork()

    inputBOW = np.random.randint(0, 2, size=10000)

    z_list, activation_list = nn.forward_pass(x=inputBOW)
    weights, biases = nn.get_weights_biases()
    print("Output probabilities:", activation_list[-1])
    print("Sum:", sum(activation_list[-1]))
    print("bias pre-backprop:", biases[-1])
    
    nn.backpropogate(z_list=z_list, activation_list=activation_list, target=np.array([1, 0, 0, 0, 0]))
    weights, biases = nn.get_weights_biases()
    print("bias post-backprop:", biases[-1])
    