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
        
    
    def create_weights_biases(self, modelSize):
        self.weights = []
        self.biases = []
        
        for i in range(0, len(modelSize)-1):
            self.weights.append(self.he_init(modelSize[i], modelSize[i+1]))
            self.biases.append(cy.zeros(modelSize[i+1]))
        
        with open("data/weights_biases.pkl", "wb") as file:
            pickle.dump({
                "weights": self.weights,
                "biases": self.biases,
            }, file)


    def load_weights_biases(self):
        with open("data/weights_biases.pkl", "rb") as file:
            data = pickle.load(file)
        
        self.weights = [cy.array(w) for w in data["weights"]]
        self.biases = [cy.array(b) for b in data["biases"]]
        
        self.m_w = [cy.zeros_like(w) for w in self.weights]
        self.m_b = [cy.zeros_like(b) for b in self.biases]
        
        self.v_w = [cy.zeros_like(w) for w in self.weights]
        self.v_b = [cy.zeros_like(b) for b in self.biases]
        
        self.t = 0
    
    
    def forward_pass(self, x, dropout_rate):
        z_list = []
        activation_list = [x]

        for i in range(len(self.weights) - 1):
            z = cy.dot(activation_list[i], self.weights[i].T) + self.biases[i]
            a = relu(z)
            
            if dropout_rate > 0:
                mask = (cy.random.rand(*a.shape) > dropout_rate).astype(a.dtype)
                a = a * mask / (1 - dropout_rate)
            
            z_list.append(z)
            activation_list.append(a)

        z_out = cy.dot(activation_list[-1], self.weights[-1].T) + self.biases[-1]
        a_out = softmax(z_out)
        z_list.append(z_out)
        activation_list.append(a_out)

        return z_list, activation_list


    def backpropagate(self, z_list, activation_list, target, batch_size, l2_lambda=0.001) -> tuple:
        gradient_weights = []
        gradient_biases = []
        
        dz_current = activation_list[-1] - target

        grad_w = cy.dot(dz_current.T, activation_list[-2]) / batch_size + l2_lambda * self.weights[-1]
        grad_b = cy.mean(dz_current, axis=0)
        
        gradient_weights.insert(0, grad_w)
        gradient_biases.insert(0, grad_b)

        for i in range(len(self.weights)-2, -1, -1):
            dz_current = cy.dot(dz_current, self.weights[i+1]) * relu_derivative(z_list[i])
            
            grad_w = cy.dot(dz_current.T, activation_list[i]) / batch_size + l2_lambda * self.weights[i]
            grad_b = cy.mean(dz_current, axis=0)
            
            gradient_weights.insert(0, grad_w)
            gradient_biases.insert(0, grad_b)
        
        return gradient_weights, gradient_biases


    def adam_optimizer(self, gradient_weights, gradient_biases, lr=0.0001, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        
        for i in range(len(self.weights)):
            self.m_w[i] = b1 * self.m_w[i] + (1-b1) * gradient_weights[i]
            self.m_b[i] = b1 * self.m_b[i] + (1-b1) * gradient_biases[i]
            
            m_w_hat = self.m_w[i] / (1 - (b1 ** self.t))
            m_b_hat = self.m_b[i] / (1 - (b1 ** self.t))
            
            self.v_w[i] = b2 * self.v_w[i] + (1 - b2) * cy.square(gradient_weights[i])
            self.v_b[i] = b2 * self.v_b[i] + (1 - b2) * cy.square(gradient_biases[i])
            
            v_w_hat = self.v_w[i] / (1 - (b2 ** self.t))
            v_b_hat = self.v_b[i] / (1 - (b2 ** self.t))
            
            self.weights[i] = self.weights[i] - lr * (m_w_hat / (cy.sqrt(v_w_hat) + eps))
            self.biases[i] = self.biases[i] - lr * (m_b_hat / (cy.sqrt(v_b_hat) + eps))


    def get_weights_biases(self) -> tuple:
        return self.weights, self.biases
    
    
    def save_weights_biases(self):
        data = {
            "weights": self.weights,
            "biases": self.biases,
        }
        
        with open(f"data/saved_models/updated_weights_biases.pkl", "wb") as file:
            pickle.dump(data, file)


    def he_init(self, fan_in, fan_out):
        return cy.random.randn(fan_out, fan_in) * cy.sqrt(2 / fan_in)


if __name__ ==  "__main__":
    nn = neuralNetwork()
    
    nn.create_weights_biases(modelSize=[784, 32, 32, 10])