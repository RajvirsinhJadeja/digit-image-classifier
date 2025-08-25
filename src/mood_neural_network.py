import numpy as np
import pickle


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator)
    return numerator/denominator


class neuralNetwork:
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize):
        """
        self.weight1 = self.he_init(inputSize, hiddenSize1)
        self.bias1 = np.zeros(hiddenSize1)

        self.weight2 = self.he_init(hiddenSize1, hiddenSize2)
        self.bias2 = np.zeros(hiddenSize2)

        self.weight3 = self.he_init(hiddenSize2, hiddenSize3)
        self.bias3 = np.zeros(hiddenSize3)

        self.weight4 = self.he_init(hiddenSize3, hiddenSize4)
        self.bias4 = np.zeros(hiddenSize4)

        self.weight5 = self.he_init(hiddenSize4, outputSize)
        self.bias5 = np.zeros(outputSize)

        with open("weights_biases.pkl", "wb") as file:
            pickle.dump({
                "weight1": self.weight1, "bias1": self.bias1,
                "weight2": self.weight2, "bias2": self.bias2,
                "weight3": self.weight3, "bias3": self.bias3,
                "weight4": self.weight4, "bias4": self.bias4,
                "weight5": self.weight5, "bias5": self.bias5
            }, file)
        
        """
        
        with open("data/weights_biases.pkl", "rb") as file:
            data = pickle.load(file)
            
        self.weight1 = np.array(data["weight1"])
        self.bias1   = np.array(data["bias1"])
        self.weight2 = np.array(data["weight2"])
        self.bias2   = np.array(data["bias2"])
        self.weight3 = np.array(data["weight3"])
        self.bias3   = np.array(data["bias3"])
        self.weight4 = np.array(data["weight4"])
        self.bias4   = np.array(data["bias4"])
        self.weight5 = np.array(data["weight5"])
        self.bias5   = np.array(data["bias5"])


    def he_init(self, fan_in, fan_out):
        return np.random.randn(fan_out, fan_in) * np.sqrt(2 / fan_in)


    def forward_pass(self, x) -> tuple:
        z1 = np.dot(self.weight1, x) + self.bias1
        a1 = relu(z1)

        z2 = np.dot(self.weight2, a1) + self.bias2
        a2 = relu(z2)

        z3 = np.dot(self.weight3, a2) + self.bias3
        a3 = relu(z3)

        z4 = np.dot(self.weight4, a3) + self.bias4
        a4 = relu(z4)

        z5 = np.dot(self.weight5, a4) + self.bias5
        a5 = softmax(z5)

        return z1, a1, z2, a2, z3, a3, z4, a4, z5, a5

"""
nn = neuralNetwork(inputSize=10000, hiddenSize1=1024, hiddenSize2=256, hiddenSize3=64, hiddenSize4=16, outputSize=5)

inputBOW = np.random.randint(0, 2, size=10000)

z1, a1, z2, a2, z3, a3, z4, a4, z5, a5 = nn.forward_pass(x=inputBOW)
print("Output probabilities:", a5)
print("Sum:", sum(a5))
"""