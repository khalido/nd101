import numpy as np

class NeuralNetwork():
    def __init__(self):
        #seed random number generator 
        np.random.seed(1)

        #model a single neuron with 3 inputs and 1 output connection 
        #assign random weights to a 3x1 matrix, with values from -1 to 1
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1
    
    #sigmoid function, which describes an s shaped curve
    #we pass the weighted sum of the inputs through this function
    #to normalize them b/w 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #pass the training set through our neural net 
            output = self.predict(training_set_inputs)

            #calculate the error 
            error = training_set_outputs - output

            #multiply error by the input and the gradient of the sigmoid curve
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #adjust the weights
            self.synaptic_weights += adjustment

    
    def predict(self, inputs):
        #pass inputs though our neural network (our single neuron)
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))



if __name__ == "__main__":
    neural_network = NeuralNetwork()
    
    print("random starting synaptic weights")
    print(neural_network.synaptic_weights)

    #the training set, consisteing of 3 input and 1 output values
    training_set_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = np.array([[0,1,1,0]]).T

    #train the neural net using the training set
    #do it 10K times and make small ajustments each time 
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("new synaptic weights after training")
    print(neural_network.synaptic_weights)
    
    #test the neural network with a new situation
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.predict(np.array([1,0,0])))
