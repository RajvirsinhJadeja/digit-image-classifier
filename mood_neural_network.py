import math
import random


class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def calc(self, inputValue):
        x = 0

        for i in range(len(inputValue)):
            x += self.weight[i] * inputValue[i]
            x += self.bias

        return max(0, x)

    def calcSoftMax(self, inputValue):
        x = 0
        temp = []
        answer = []

        for i in range(len(inputValue)):
            x += math.exp(inputValue[i])
            temp.append(math.exp(inputValue[i]))

        for num in temp:
            answer.append(num / x)

        return answer


Top1000Vocab = []
inputBOW = [random.randint(0, 1) for _ in range(1000)]

weight1 = [
    [random.uniform(-(math.sqrt(2 / 1000)), (math.sqrt(2 / 1000))) for _ in range(1000)]
    for _ in range(32)
]
bias1 = [0] * 32

weight2 = [
    [random.uniform(-(math.sqrt(2 / 32)), (math.sqrt(2 / 32))) for _ in range(32)]
    for _ in range(16)
]
bias2 = [0] * 16

weight3 = [
    [random.uniform(-(math.sqrt(2 / 16)), (math.sqrt(2 / 16))) for _ in range(16)]
    for _ in range(5)
]
bias3 = [0] * 5

output = []
for i in range(0, 32):
    n = Neuron(weight1[i], bias1[i])
    output.append(n.calc(inputBOW))

print(output)
print(len(output))

output2 = []
for i in range(0, 16):
    n = Neuron(weight2[i], bias2[i])
    output2.append(n.calc(output))

print(output2)
print(len(output2))

output3 = []
for i in range(0, 5):
    n = Neuron(weight3[i], bias3[i])
    output3.append(n.calc(output2))

print(output3)
print(len(output3))

n = Neuron(weight3[0], bias3[0])
print("\n")
print(n.calcSoftMax(output3))
