import cupy as cy
import pickle


def relu(x):
    return cy.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    e_x = cy.exp(x - cy.max(x, axis=1, keepdims=True))
    return e_x / cy.sum(e_x, axis=1, keepdims=True)


class neuralNetwork:
    def __init__(self):
        return
        
    
    def create_weights_biases(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, outputSize):
        self.weight1 = self.he_init(inputSize, hiddenSize1)
        self.bias1 = cy.zeros(hiddenSize1)

        self.weight2 = self.he_init(hiddenSize1, hiddenSize2)
        self.bias2 = cy.zeros(hiddenSize2)

        self.weight3 = self.he_init(hiddenSize2, hiddenSize3)
        self.bias3 = cy.zeros(hiddenSize3)

        self.weight4 = self.he_init(hiddenSize3, outputSize)
        self.bias4 = cy.zeros(outputSize)

        with open("data/weights_biases.pkl", "wb") as file:
            pickle.dump({
                "weight1": self.weight1, "bias1": self.bias1,
                "weight2": self.weight2, "bias2": self.bias2,
                "weight3": self.weight3, "bias3": self.bias3,
                "weight4": self.weight4, "bias4": self.bias4,
            }, file)


    def load_weights_biases(self):
        with open("data/weights_biases.pkl", "rb") as file:
            data = pickle.load(file)
        
        self.weights = [cy.array(data["weight1"]), cy.array(data["weight2"]), cy.array(data["weight3"]), cy.array(data["weight4"])]
        self.biases = [cy.array(data["bias1"]), cy.array(data["bias2"]), cy.array(data["bias3"]), cy.array(data["bias4"])]
    
    
    def forward_pass(self, x) -> tuple:
        z_list = []
        activation_list = [x]
        
        for i in range(0, len(self.weights) - 1):
            z = cy.dot(activation_list[i], self.weights[i].T) + self.biases[i]
            a = relu(z)
            
            z_list.append(z)
            activation_list.append(a)

        z5 = cy.dot(activation_list[-1], self.weights[-1].T) + self.biases[-1]
        a5 = softmax(z5)
        
        z_list.append(z5)
        activation_list.append(a5)

        return z_list, activation_list
    
    
    def backpropogate(self, z_list, activation_list, target, batch_size, lr=0.001):
        updated_weights = []
        updated_biases = []
        
        dz_list = [activation_list[-1] - target]
        
        updated_weights.append(self.weights[-1] - (lr * (cy.dot(dz_list[-1].T, activation_list[-2]) /  batch_size)))
        updated_biases.append(self.biases[-1] - (lr * cy.mean(dz_list[-1], axis=0)))
        
        for i in range(len(self.weights)-2, -1, -1):
             
        
        self.weights = updated_weights
        self.biases = updated_biases


    def get_weights_biases(self) -> tuple:
        return self.weights, self.biases
    
    
    def save_weights_biases(self, number):
        data = {
            "weight1": self.weights[0],
            "weight2": self.weights[1],
            "weight3": self.weights[2],
            "weight4": self.weights[3],
            "bias1": self.biases[0],
            "bias2": self.biases[1],
            "bias3": self.biases[2],
            "bias4": self.biases[3],
        }
        
        with open(f"data/saved_models/weights_biases_epoch{number}.pkl", "wb") as file:
            pickle.dump(data, file)


    def he_init(self, fan_in, fan_out):
        return cy.random.randn(fan_out, fan_in) * cy.sqrt(2 / fan_in)


if __name__ ==  "__main__":
    nn = neuralNetwork()
    nn.create_weights_biases(inputSize=100, hiddenSize1=32, hiddenSize2=16, hiddenSize3=8, outputSize=5)
    
    """
    icyutBOW = cy.random.randint(0, 2, size=10000)

    z_list, activation_list = nn.forward_pass(x=icyutBOW)
    weights, biases = nn.get_weights_biases()
    print("Output probabilities:", activation_list[-1])
    print("Sum:", sum(activation_list[-1]))
    print("bias pre-backprop:", biases[-1])
    
    nn.backpropogate(z_list=z_list, activation_list=activation_list, target=cy.array([1, 0, 0, 0, 0]))
    weights, biases = nn.get_weights_biases()
    print("bias post-backprop:", biases[-1])
    """