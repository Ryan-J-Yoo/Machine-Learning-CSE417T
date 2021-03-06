• Generate an 11-dimensional weight vector w∗, where the first dimension is 0 and the other 10 dimensions are sampled independently at random from the uniform (0, 1) distribution (the first dimension will serve as the threshold and we set it to 0 for convenience).

• Generate a random training set with 100 examples, where each dimension of each train- ing example is sampled independently at random from the uniform (-1, 1) distribution. The examples are all classified in accordance with w∗.

• Run the perceptron learning algorithm, starting with a zero weight vector, on the training set you just generated. Keep track of the number of iterations it takes to learn a hypothesis that correctly separates the training data.
