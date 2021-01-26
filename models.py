import nn

class PerceptronModel(object):
	def __init__(self, dimensions):
		"""
		Initialize a new Perceptron instance.

		A perceptron classifies data points as either belonging to a particular
		class (+1) or not (-1). `dimensions` is the dimensionality of the data.
		For example, dimensions=2 would mean that the perceptron must classify
		2D points.
		"""
		self.w = nn.Parameter(1, dimensions)

	def get_weights(self):
		"""
		Return a Parameter instance with the current weights of the perceptron.
		"""
		return self.w

	def run(self, x):
		"""
		Calculates the score assigned by the perceptron to a data point x.

		Inputs:
			x: a node with shape (1 x dimensions)
		Returns: a node containing a single number (the score)
		"""
		"*** YOUR CODE HERE ***"

		w = self.get_weights()
		return nn.DotProduct(w, x)

	def get_prediction(self, x):
		"""
		Calculates the predicted class for a single data point `x`.

		Returns: 1 or -1
		"""
		"*** YOUR CODE HERE ***"

		score = nn.as_scalar(self.run(x))
		return 1 if score >= 0 else -1

	def train(self, dataset):
		"""
		Train the perceptron until convergence.
		"""
		"*** YOUR CODE HERE ***"

		converge = True

		for x, y in dataset.iterate_once(1):
			if (self.get_prediction(x) == nn.as_scalar(y)): continue
			converge = False
			self.w.update(x, nn.as_scalar(y))

		if (converge == True): return None
		else: return self.train(dataset)

class RegressionModel(object):
	"""
	A neural network model for approximating a function that maps from real
	numbers to real numbers. The network should be sufficiently large to be able
	to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
	"""
	def __init__(self):
		# Initialize your model parameters here
		"*** YOUR CODE HERE ***"

		self.multiplier = -0.1  # has to be negative to work
		self.batch_size = 200
		self.layer1_size = 20
		self.layer2_size = 20

		# 2 hidden layer
		self.m1 = nn.Parameter(1, self.layer1_size)
		self.b1 = nn.Parameter(1, self.layer1_size)
		self.m2 = nn.Parameter(self.layer1_size, self.layer2_size)
		self.b2 = nn.Parameter(1, self.layer2_size)
		self.m3 = nn.Parameter(self.layer2_size, 1)
		self.b3 = nn.Parameter(1, 1)

	def run(self, x):
		"""
		Runs the model for a batch of examples.

		Inputs:
			x: a node with shape (batch_size x 1)
		Returns:
			A node with shape (batch_size x 1) containing predicted y-values
		"""
		"*** YOUR CODE HERE ***"

		# f(x) = relu(relu(x * w1 + b1) * w2 + b2) * w3 + b3

		temp = nn.Linear(x, self.m1)
		temp = nn.AddBias(temp, self.b1)
		temp = nn.ReLU(temp)
		temp = nn.Linear(temp, self.m2)
		temp = nn.AddBias(temp, self.b2)
		temp = nn.ReLU(temp)
		temp = nn.Linear(temp, self.m3)
		predicted_y = nn.AddBias(temp, self.b3)
		
		return predicted_y


	def get_loss(self, x, y):
		"""
		Computes the loss for a batch of examples.

		Inputs:
			x: a node with shape (batch_size x 1)
			y: a node with shape (batch_size x 1), containing the true y-values
				to be used for training
		Returns: a loss node
		"""
		"*** YOUR CODE HERE ***"

		predicted_y = self.run(x)
		loss = nn.SquareLoss(predicted_y, y)

		return loss

	def train(self, dataset):
		"""
		Trains the model.
		"""
		"*** YOUR CODE HERE ***"

		converge = False

		while not converge:

			for x, y in dataset.iterate_once(self.batch_size):
				
				grad_wrt_v = nn.gradients(self.get_loss(x, y), [self.m1, self.b1, self.m2, self.b2, self.m3, self.b3])
				self.m1.update(grad_wrt_v[0], self.multiplier)
				self.b1.update(grad_wrt_v[1], self.multiplier)
				self.m2.update(grad_wrt_v[2], self.multiplier)
				self.b2.update(grad_wrt_v[3], self.multiplier)
				self.m3.update(grad_wrt_v[4], self.multiplier)
				self.b3.update(grad_wrt_v[5], self.multiplier)

			if (nn.as_scalar(self.get_loss(x, y)) <= 0.02): 
				converge = True


class DigitClassificationModel(object):
	"""
	A model for handwritten digit classification using the MNIST dataset.

	Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
	into a 784-dimensional vector for the purposes of this model. Each entry in
	the vector is a floating point number between 0 and 1.

	The goal is to sort each digit into one of 10 classes (number 0 through 9).

	(See RegressionModel for more information about the APIs of different
	methods here. We recommend that you implement the RegressionModel before
	working on this part of the project.)
	"""
	def __init__(self):
		# Initialize your model parameters here
		"*** YOUR CODE HERE ***"

		self.multiplier = -0.5   # has to be negative to work
		self.batch_size = 500
		self.layer1_size = 200
		self.layer2_size = 50

		# 2 hidden layer
		self.m1 = nn.Parameter(784, self.layer1_size)
		self.b1 = nn.Parameter(1, self.layer1_size)
		self.m2 = nn.Parameter(self.layer1_size, self.layer2_size)
		self.b2 = nn.Parameter(1, self.layer2_size)
		self.m3 = nn.Parameter(self.layer2_size, 10)
		self.b3 = nn.Parameter(1, 10)

	def run(self, x):
		"""
		Runs the model for a batch of examples.

		Your model should predict a node with shape (batch_size x 10),
		containing scores. Higher scores correspond to greater probability of
		the image belonging to a particular class.

		Inputs:
			x: a node with shape (batch_size x 784)
		Output:
			A node with shape (batch_size x 10) containing predicted scores
				(also called logits)
		"""
		"*** YOUR CODE HERE ***"

		# f(x) = relu(relu(x * w1 + b1) * w2 + b2) * w3 + b3

		temp = nn.Linear(x, self.m1)
		temp = nn.AddBias(temp, self.b1)
		temp = nn.ReLU(temp)
		temp = nn.Linear(temp, self.m2)
		temp = nn.AddBias(temp, self.b2)
		temp = nn.ReLU(temp)
		temp = nn.Linear(temp, self.m3)
		predicted_y = nn.AddBias(temp, self.b3)
		
		return predicted_y

	def get_loss(self, x, y):
		"""
		Computes the loss for a batch of examples.

		The correct labels `y` are represented as a node with shape
		(batch_size x 10). Each row is a one-hot vector encoding the correct
		digit class (0-9).

		Inputs:
			x: a node with shape (batch_size x 784)
			y: a node with shape (batch_size x 10)
		Returns: a loss node
		"""
		"*** YOUR CODE HERE ***"

		predicted_y = self.run(x)
		loss = nn.SoftmaxLoss(predicted_y, y)

		return loss

	def train(self, dataset):
		"""
		Trains the model.
		"""
		"*** YOUR CODE HERE ***"

		converge = False

		while not converge:

			for x, y in dataset.iterate_once(self.batch_size):

				grad_wrt_v = nn.gradients(self.get_loss(x, y), [self.m1, self.b1, self.m2, self.b2, self.m3, self.b3])
				self.m1.update(grad_wrt_v[0], self.multiplier)
				self.b1.update(grad_wrt_v[1], self.multiplier)
				self.m2.update(grad_wrt_v[2], self.multiplier)
				self.b2.update(grad_wrt_v[3], self.multiplier)
				self.m3.update(grad_wrt_v[4], self.multiplier)
				self.b3.update(grad_wrt_v[5], self.multiplier)

			if (dataset.get_validation_accuracy() >= 0.97): 
				converge = True

class LanguageIDModel(object):
	"""
	A model for language identification at a single-word granularity.

	(See RegressionModel for more information about the APIs of different
	methods here. We recommend that you implement the RegressionModel before
	working on this part of the project.)
	"""
	def __init__(self):
		# Our dataset contains words from five different languages, and the
		# combined alphabets of the five languages contain a total of 47 unique
		# characters.
		# You can refer to self.num_chars or len(self.languages) in your code
		self.num_chars = 47
		self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

		# Initialize your model parameters here
		"*** YOUR CODE HERE ***"

	def run(self, xs):
		"""
		Runs the model for a batch of examples.

		Although words have different lengths, our data processing guarantees
		that within a single batch, all words will be of the same length (L).

		Here `xs` will be a list of length L. Each element of `xs` will be a
		node with shape (batch_size x self.num_chars), where every row in the
		array is a one-hot vector encoding of a character. For example, if we
		have a batch of 8 three-letter words where the last word is "cat", then
		xs[1] will be a node that contains a 1 at position (7, 0). Here the
		index 7 reflects the fact that "cat" is the last word in the batch, and
		the index 0 reflects the fact that the letter "a" is the inital (0th)
		letter of our combined alphabet for this task.

		Your model should use a Recurrent Neural Network to summarize the list
		`xs` into a single node of shape (batch_size x hidden_size), for your
		choice of hidden_size. It should then calculate a node of shape
		(batch_size x 5) containing scores, where higher scores correspond to
		greater probability of the word originating from a particular language.

		Inputs:
			xs: a list with L elements (one per character), where each element
				is a node with shape (batch_size x self.num_chars)
		Returns:
			A node with shape (batch_size x 5) containing predicted scores
				(also called logits)
		"""
		"*** YOUR CODE HERE ***"

	def get_loss(self, xs, y):
		"""
		Computes the loss for a batch of examples.

		The correct labels `y` are represented as a node with shape
		(batch_size x 5). Each row is a one-hot vector encoding the correct
		language.

		Inputs:
			xs: a list with L elements (one per character), where each element
				is a node with shape (batch_size x self.num_chars)
			y: a node with shape (batch_size x 5)
		Returns: a loss node
		"""
		"*** YOUR CODE HERE ***"

	def train(self, dataset):
		"""
		Trains the model.
		"""
		"*** YOUR CODE HERE ***"
