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
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, outputSize):
        self.weight1 = [
            [random.uniform(-math.sqrt(2 / inputSize), math.sqrt(2 / inputSize))
            for _ in range(inputSize)]
            for _ in range(hiddenSize1)
        ]
        self.bias1 = [0 for _ in range(hiddenSize1)]

        self.weight2 = [
            [random.uniform(-math.sqrt(2 / hiddenSize1), math.sqrt(2 / hiddenSize1))
            for _ in range(hiddenSize1)]
            for _ in range(hiddenSize2)
        ]
        self.bias2 = [0 for _ in range(hiddenSize2)]

        self.weight3 = [
            [random.uniform(-math.sqrt(2 / hiddenSize2), math.sqrt(2 / hiddenSize2))
            for _ in range(hiddenSize2)]
            for _ in range(hiddenSize3)
        ]
        self.bias3 = [0 for _ in range(hiddenSize3)]
        
        self.weight4 = [[random.uniform(-math.sqrt(2 / hiddenSize3), math.sqrt(2 / hiddenSize3)) for _ in range(hiddenSize3)] for _ in range(outputSize)]
        self.bias4 = [0 for _ in range(outputSize)]
        
        print("\nLength of weight1[0]: ", len(self.weight1[0]))
        print("Length of weight1: ", len(self.weight1))
        
        print("\nLength of weight2[0]: ", len(self.weight2[0]))
        print("Length of weight2: ", len(self.weight2))
        
        print("\nLength of weight3[0]: ", len(self.weight3[0]))
        print("Length of weight3: ", len(self.weight3))
        
        print("\nLength of weight4[0]: ", len(self.weight4[0]))
        print("Length of weight4: ", len(self.weight4))

    def forward(self, x):
        self.hidden1 = [
            relu(sum(w * i for w, i in zip(weights, x)) + bias)
            for weights, bias in zip(self.weight1, self.bias1)
        ]

        self.hidden2 = [
            relu(sum(w * i for w, i in zip(weights, self.hidden1)) + bias)
            for weights, bias in zip(self.weight2, self.bias2)
        ]

        self.hidden3 = [
            relu(sum(w * i for w, i in zip(weights, self.hidden2)) + bias)
            for weights, bias in zip(self.weight3, self.bias3)
        ]
        
        self.output = [
            sum(w * i for w, i in zip(weights, self.hidden3)) + bias
            for weights, bias in zip(self.weight4, self.bias4)
        ]
        
        return softmax(self.output)


nn = neuralNetwork(inputSize=1000, hiddenSize1=64, hiddenSize2=32, hiddenSize3=16, outputSize=5)

inputBOW = [random.randint(0, 1) for _ in range(1000)]

outputProbability = nn.forward(x=inputBOW)
print("Output probabilities:", outputProbability)
print("Sum:", sum(outputProbability))
