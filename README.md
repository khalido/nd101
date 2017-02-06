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
- [Linear algebra youtube playlist](https://www.youtube.com/playlist?list=PLlXfTHzgMRUKXD88IdzS14F4NxAZudSmv)
- [Khan Academy's Linear Algebra Intro](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/jacobian/v/jacobian-prerequisite-knowledge)

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

### Books to read

- [Grokking Deep Learning by Andrew Trask](https://www.manning.com/books/grokking-deep-learning). This provides a very gentle introduction to Deep Learning and covers the intuition more than the theory.
- [Neural Networks And Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen. This book is more rigorous than Grokking Deep Learning and includes a lot of fun, interactive visualizations to play with.
- [The Deep Learning Textbook](http://www.deeplearningbook.org/) from Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This online book has lot of material and is the most rigorous of the three books suggested.

### [Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird)

This is a deep learning agent which plays flappy bird. Not very useful at this point. Need to know to understand/use/train the flappybird playing agent!

So come back to this later.

### Siraj's intro to linear regression

he walks through linear regression, using numpy and then by building a gradient descent model.


### moving on to neural nets

Watch this [Backpropagation video](https://www.youtube.com/watch?v=i94OvYb6noo)

- a neural network is an algorithm which identifies patterns in data
- Backpropagation trains a neural net by updating weights via gradient descent
- deep learning = many layer neural net + big data + big compute



