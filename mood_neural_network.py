import math
import random
import pickle


def relu(x):
    return max(0, x)


def softmax(list):
    expList = [math.exp(x) for x in list]
    total = sum(expList)
    return [x / total for x in expList]


class neuralNetwork:
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize):
        """
        self.weight1 = [
            [random.uniform(-math.sqrt(2 / inputSize), math.sqrt(2 / inputSize)) for _ in range(inputSize)]
            for _ in range(hiddenSize1)
        ]
        self.bias1 = [0 for _ in range(hiddenSize1)]

        self.weight2 = [
            [random.uniform(-math.sqrt(2 / hiddenSize1), math.sqrt(2 / hiddenSize1)) for _ in range(hiddenSize1)]
            for _ in range(hiddenSize2)
        ]
        self.bias2 = [0 for _ in range(hiddenSize2)]

        self.weight3 = [
            [random.uniform(-math.sqrt(2 / hiddenSize2), math.sqrt(2 / hiddenSize2)) for _ in range(hiddenSize2)]
            for _ in range(hiddenSize3)
        ]
        self.bias3 = [0 for _ in range(hiddenSize3)]

        self.weight4 = [
            [random.uniform(-math.sqrt(2 / hiddenSize3), math.sqrt(2 / hiddenSize3)) for _ in range(hiddenSize3)]
            for _ in range(hiddenSize4)
        ]
        self.bias4 = [0 for _ in range(hiddenSize4)]

        self.weight5 = [
            [random.uniform(-math.sqrt(2 / hiddenSize4), math.sqrt(2 / hiddenSize4)) for _ in range(hiddenSize4)]
            for _ in range(outputSize)
        ]
        self.bias5 = [0 for _ in range(outputSize)]

        with open("weights_biases.pkl", "wb") as file:
            pickle.dump({
                "weight1": self.weight1, "bias1": self.bias1,
                "weight2": self.weight2, "bias2": self.bias2,
                "weight3": self.weight3, "bias3": self.bias3,
                "weight4": self.weight4, "bias4": self.bias4,
                "weight5": self.weight5, "bias5": self.bias5
            }, file)
        
        """
        
        with open("weights_biases.pkl", "rb") as file:
            data = pickle.load(file)
            self.weight1 = data["weight1"]
            self.bias1 = data["bias1"]
            self.weight2 = data["weight2"]
            self.bias2 = data["bias2"]
            self.weight3 = data["weight3"]
            self.bias3 = data["bias3"]
            self.weight4 = data["weight4"]
            self.bias4 = data["bias4"]
            self.weight5 = data["weight5"]
            self.bias5 = data["bias5"]


    def forward_pass(self, x) -> tuple:
        z1 = [sum(w * i for w, i in zip(weights, x)) + b for weights, b in zip(self.weight1, self.bias1)]
        a1 = [relu(z) for z in z1]

        z2 = [sum(w * i for w, i in zip(weights, a1)) + b for weights, b in zip(self.weight2, self.bias2)]
        a2 = [relu(z) for z in z2]

        z3 = [sum(w * i for w, i in zip(weights, a2)) + b for weights, b in zip(self.weight3, self.bias3)]
        a3 = [relu(z) for z in z3]

        z4 = [sum(w * i for w, i in zip(weights, a3)) + b for weights, b in zip(self.weight4, self.bias4)]
        a4 = [relu(z) for z in z4]

        z5 = [sum(w * i for w, i in zip(weights, a4)) + b for weights, b in zip(self.weight5, self.bias5)]
        a5 = softmax(z5)

        return z1, a1, z2, a2, z3, a3, z4, a4, z5, a5

"""
nn = neuralNetwork(inputSize=10000, hiddenSize1=1024, hiddenSize2=256, hiddenSize3=64, hiddenSize4=16, outputSize=5)

inputBOW = [random.randint(0, 1) for _ in range(10000)]

z1, a1, z2, a2, z3, a3, z4, a4, z5, a5 = nn.forward_pass(x=inputBOW)
print("Output probabilities:", a5)
print("Sum:", sum(a5))
"""