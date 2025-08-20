import pickle

with open("initial_Weights_Biases.pkl", "rb") as file:
            data = pickle.load(file)
            weight1 = data["weight1"]
            bias1 = data["bias1"]
            weight2 = data["weight2"]
            bias2 = data["bias2"]
            weight3 = data["weight3"]
            bias3 = data["bias3"]
            weight4 = data["weight4"]
            bias4 = data["bias4"]

input = []
a1 = []
a2 = []
a3 = [0.5, 0.1, 0.0063374037529112595, 0, 0.24047889232436315, 0, 0.10121426343074791, 0.12380663921879813, 0.01261093472213534, 0, 0.06075538014272358, 0, 0.07477807304462604, 0.05495819042854164, 0, 0.19670778705749647]
            
predicted = [0.19, 0.18, 0.18, 0.22, 0.20]
target = [1, 0, 0, 0, 0]

dL_dz4 = [x - y for x, y in zip(predicted, target)]
print("dL_dz4: ", dL_dz4)

dL_dw4 = [[x * y for y in a3] for x in dL_dz4]

new_w4 = [[w - (slope * 0.1) for w, slope in zip(weight_row, grad_row)] for weight_row, grad_row in zip(weight4, dL_dw4)]
new_b4 = [old_bias - (slope * 0.1) for old_bias, slope in zip(bias4, dL_dz4)]

dL_da3 = [[dz * w for w in row] for dz, row in zip(dL_dz4, weight4)]