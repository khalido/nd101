# nd101

Notes & files for [Udacity's Deep Learning intro course](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101).

## Before the course starts:

- Read [Machine learning is fun](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471#.t1zhql96j)
- Watch [Andrew NG: Deep Learning in Practice](https://youtu.be/LFDU2GX4AqM) (34 minutes)
- Go though [UD730 deep learning course](https://classroom.udacity.com/courses/ud730/) on Udacity
- Watch [Learn tensorflow 3 hour video](https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd)

### Python resources

The course recommneds knowing [basic python from here](https://www.udacity.com/course/programming-foundations-with-python--ud036), but I found the following two resources better:

- [A whirlwind tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython)
- [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook)

### Math resources

Need to know multivariable calculus & linear algebra.
 
- [Khan Academy Multivariable calculus](https://www.khanacademy.org/math/multivariable-calculus)
- [Linear algebra youtube playlist](https://www.youtube.com/playlist?list=PLlXfTHzgMRUKXD88IdzS14F4NxAZudSmv) or over at [lemma](https://www.lem.ma/web/#/books/VBS92YDYuscc5-lK/landing)
- [Khan Academy's Linear Algebra Intro](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/jacobian/v/jacobian-prerequisite-knowledge)

### Books to read

- [Grokking Deep Learning by Andrew Trask](https://www.manning.com/books/grokking-deep-learning). This provides a very gentle introduction to Deep Learning and covers the intuition more than the theory.
- [Neural Networks And Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen. This book is more rigorous than Grokking Deep Learning and includes a lot of fun, interactive visualizations to play with.
- [The Deep Learning Textbook](http://www.deeplearningbook.org/) from Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This online book has lot of material and is the most rigorous of the three books suggested.

# Week 1

The two instructors are [Mat Leonard] & [Siraj Raval](http://www.sirajraval.com/).

Some of the stuff covered in the first week:

- [Scikit-learn](http://scikit-learn.org/) - An extremely popular Machine Learning library for python.
- [Perceptrons](https://en.wikipedia.org/wiki/Perceptron) -The simplest form of a neural network.
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) - A process by which Machine Learning algorithms learn to improve themselves based on the accuracy of their predictions. You'll learn more about this in upcoming lessons.
- [Backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html) - The process by which neural networks learn how to improve individual parameters. You'll learn all about this in the upcoming lessons.
- [Numpy](http://www.numpy.org/) - An extremely popular library for scientific computing in python.
- [Tensorflow](http://tensorflow.org/) - One of the most popular python libraries for creating neural networks. It is maintained by Google.

### fast-style transfer

Looking at[an existing style transfer deep learning script](https://github.com/lengstrom/fast-style-transfer) to play around with. Hmm.. interesting to see what can be done but HOW is it done is the q?

### [Deep Traffic](http://selfdrivingcars.mit.edu/deeptrafficjs/) simulator

See the [overview](http://selfdrivingcars.mit.edu/deeptraffic/) for how to tinker with the inputs to train the simple neural network. Interesting to see how inputs drastically effect the quality of the output.

> [DeepTraffic](http://selfdrivingcars.mit.edu/deeptrafficjs/) is a gamified simulation of typical highway traffic. Your task is to build a neural agent – more specifically design and train a neural network that performs well on high traffic roads. Your neural network gets to control one of the cars (displayed in red) and has to learn how to navigate efficiently to go as fast as possible. The car already comes with a safety system, so you don’t have to worry about the basic task of driving – the net only has to tell the car if it should accelerate/slow down or change lanes, and it will do so if that is possible without crashing into other cars.

This neural net uses [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) 

### [Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird)

This is a deep learning agent which plays flappy bird. Not very useful at this point. Need to know to understand/use/train the flappybird playing agent!

So come back to this later.

### Siraj's intro to linear regression

he walks through linear regression, using numpy and then by building a gradient descent model.


### moving on to neural nets

- a neural network is an algorithm which identifies patterns in data
- Backpropagation trains a neural net by updating weights via gradient descent
- deep learning = many layer neural net + big data + big compute

## Backpropagation

This is the key to understanding neural nets, so it's important to understand how Backpropagation works. 

- [CS231n Winter 2016 Lecture 4: Backpropagation](https://www.youtube.com/watch?v=59Hbtz7XgjM)


## Project 1: Predictin Bike Sharing demand from historical data

[Final Notebook](https://github.com/khalido/nd101-projects/blob/playing-with-numpy/dlnd-your-first-neural-network.ipynb)

# Week 2

## Model Evaluation and Validation

Generalization is better than overfitting. 

R2 score
- simplest possible model is to take the avg of all values and draw a straight line, then calculate the mean square error
 - the R2 score is 1 minus the error of our regression model divided by the error of the simplest possible model
 - if we have a good model, the error will be small compared to the simple model, thus R2 will be close to 1
 - for a bad model, the ratio of errors will be closer to 1, giving a small R2 values

 from sklearn.metrics import r2_score

### Two types of error

- overfitting
- underfitting


# Week 3

## Sentiment Analysis with Andrew Trask

This was a good project - built a simple neural network to classify reviews as negative or positive.

## Intro to TFLearn

Sigmoid activation functions have a max derivate of .25, so errors shrink by 75% or more during backprogation. This means the neural network takes a long time to train. Instead of sigmoid, most DL networks use RLU - which is a supersimple function which outputs max(input, 0). For a +ve inut the output equals the input, and for a -ve input the output is 0. Relu nodes can die if there is a large graident through them, so they are best used with a small learning rate.

For a simple binary classification, the sigmoid function works, but for mulitple outputs, for example reconigizing digits, use the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function. A softmax function squashes outputs to be b/w 0 and 1 and divides them such that the total sum of the output equals 1.

one hot encoding means using a simple vector, like `y=[0,0,0,0,1,0,0,0,0,0]` to represent 4. 

http://tflearn.org/

## RNN

http://colah.github.io/posts/2015-08-Understanding-LSTMs/
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

## using tflearn to do sentiment Analysis


# Week 4

## Siraj's Math for Deep learning
Need to know statistics, linear algebra and calculus. 
C

