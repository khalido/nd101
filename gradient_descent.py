import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1 - collect our data
df = pd.read_csv('data/live_reg_data.csv', header=None)
print(df.head())

#collect data using numpy
points = np.genfromtxt('data/live_reg_data.csv', delimiter=',')
points[:5]

#lets see the data
plt.scatter(df[0], df[1])
plt.show()

#parameters
learning_rate = 0.0001
initial_b = 0
initial_m = 0
num_iterations = 1000

def compute_error_for_line_given_points(b, m , points):
    totalError = 0 #initialize error at 0
    for i in range(0, len(points)): #for every point
        x = points[i, 0] #get x val
        y = points[i, 1] #get y val
        totalError += (y - (m*x + b)) **2
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    
    #gradient descent
    for i in range(num_iterations):
        #update b & m with new more accurate b and m
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b,m]

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #direction with respect to b and m
        #computing partial deriavitives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
print('ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
