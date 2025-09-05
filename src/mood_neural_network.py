import cupy as cy
import pickle


def relu(x):
    return cy.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    numerator = cy.exp(x - cy.max(x))
    return numerator/cy.sum(numerator)


class neuralNetwork:
    def __init__(self):
        with open("data/weights_biases.pkl", "rb") as file:
            data = pickle.load(file)
        
        self.weights = [cy.array(data["weight1"]), cy.array(data["weight2"]), cy.array(data["weight3"]), cy.array(data["weight4"]), cy.array(data["weight5"])]
        self.biases = [cy.array(data["bias1"]), cy.array(data["bias2"]), cy.array(data["bias3"]), cy.array(data["bias4"]), cy.array(data["bias5"])]
    
    
    def forward_pass(self, x) -> tuple:
        z_list = []
        activation_list = [x]
        
        for i in range(len(self.weights) - 1):
            z = cy.dot(self.weights[i], activation_list[i]) + self.biases[i]
            a = relu(z)
            
            z_list.append(z)
            activation_list.append(a)

        z5 = cy.dot(self.weights[-1], activation_list[-1]) + self.biases[-1]
        a5 = softmax(z5)
        
        z_list.append(z5)
        activation_list.append(a5)

        return z_list, activation_list
    
    
    def backpropogate(self, z_list, activation_list, target, lr=0.01):
        updated_weights = []
        updated_biases = []
        
        dz_list = [activation_list[-1] - target]
        
        updated_weights.append(self.weights[-1] - (lr * (cy.outer(dz_list[-1], activation_list[-2]))))
        updated_biases.append(self.biases[-1] - (lr * dz_list[-1]))
        
        for i in range(4, 0, -1):
            dz = cy.dot(self.weights[i].T, dz_list[-1]) * relu_derivative(z_list[i-1])
            
            updated_weights.insert(0, self.weights[i-1] - (lr * (cy.outer(dz, activation_list[i-1]))))
            updated_biases.insert(0, self.biases[i-1] - (lr * dz))
            
            dz_list.append(dz)
        
        self.weights = updated_weights
        self.biases = updated_biases


    def get_weights_biases(self) -> tuple:
        return self.weights, self.biases
    
    
    def set_weights_biases(self, number):
        data = {
            "weight1": self.weights[0],
            "weight2": self.weights[1],
            "weight3": self.weights[2],
            "weight4": self.weights[3],
            "weight5": self.weights[4],
            "bias1": self.biases[0],
            "bias2": self.biases[1],
            "bias3": self.biases[2],
            "bias4": self.biases[3],
            "bias5": self.biases[4],
        }
        
        with open(f"data/saved_models/weights_biases_epoch{number}.pkl", "wb") as file:
            pickle.dump(data, file)


if __name__ ==  "__main__":
    nn = neuralNetwork()

    icyutBOW = cy.random.randint(0, 2, size=10000)

    z_list, activation_list = nn.forward_pass(x=icyutBOW)
    weights, biases = nn.get_weights_biases()
    print("Output probabilities:", activation_list[-1])
    print("Sum:", sum(activation_list[-1]))
    print("bias pre-backprop:", biases[-1])
    
    nn.backpropogate(z_list=z_list, activation_list=activation_list, target=cy.array([1, 0, 0, 0, 0]))
    weights, biases = nn.get_weights_biases()
    print("bias post-backprop:", biases[-1])
    