import tensorflow as tf
import numpy as np
import batcher as bt

def initWeightAndBias(M1, M2):
	"""
	:param M1: number of input nodes in the layer
	:param M2: number of output nodes in th layer
	:return: a weight matrix and a bias unit vector
	"""

	print("initWeightAndBias()")
	# Fill a M1*M2 matrix with random values chosen on a gaussian
	# distrobution. Divide by a number to keep the values small
	# I don't know why we divide by sqrt of m1 and not some other number thoug
	W = np.random.randn(M1,M2) / np.sqrt(M1)

	# Initialize bias to a M2-sized origin vector
	b = np.zeros(M2)

	print("return W.astype(np.float32), b.astype(np.float32)")
	# Make all values numpy float so tensorflow doesn't whine
	return W.astype(np.float32), b.astype(np.float32)

def initFilter(shape,poolsize):
	"""
	:param shape: the shape of the filter for a cnn.
		By convention it should be a 4D matrix the  first two
		values are the filter width and height. The next two
		values are the input and output channels
	:param poolsize: the factor by which we reduce the image after filtering it
	:return: a weight matrix
	"""

	print("initFilter()")

	# *shape I believe unpacks the tupled values in shape
	# fw,fh,chi,cho = shape, p1, p2 = poolsize
	# np.prod(shape[:-1]) = fw*fh*chi
	# as with initWeightAndBias() we divide the values by the sqrt of the input size
	W = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod( shape[:-2]/np.prod(poolsize) ))

	# cho * (fw*fh/(p1*p2)) =? cho*fw*fh/p1/p2
	#W = W + shape[-1]*np.prod( shape[:-2]/np.prod(poolsize))

	# Again convert to np.float32 so tf doesn't whine to us
	return W.astype(np.float32)

# Fully connected hidden layer for use in tensorflow
class DenseLayer(object):
	def __init__(self,m1,m2, my_id):
		# I don't know why dense layer has an ID but
		#convpool doesn't
		print("init dense layer")
		self.id = my_id
		self.M1 = m1
		self.M2 = m2

		# weight and bias of this layer
		W, b = initWeightAndBias(self.M1,self.M2)
		print("weight and bias done")
		self.W = tf.Variable(W)
		self.b = tf.Variable(b)

		print("params set")
		#keep params here to make getting them later easy
		self.params = [self.W, self.b]

	def forward(self, X):
		"""
		:param X: tf.Variable in the form of a (n,m1)-sized
			matrix representing the data to pass through
			the layer
		:return: tensorflow expression. amounts to relu(X*W+b)
		"""
		return tf.nn.relu( tf.matmul(X,self.W)+self.b )


class ConvPoolLayer(object):
	def __init__(self,chi,cho,fw=5,fh=5,poolSz=(2,2)):
		print("init convpoollayer")
		# init filter matrix
		sz = (fw,fh, chi,cho)
		Weight = initFilter(sz,poolSz)
		self.W = tf.Variable(Weight)

		print("set zeros")
		# init bias vector
		# it is a cho-dimensional vector where cho is the
		# number of output channels
		bias = np.zeros(cho, dtype=np.float32)
		self.b = tf.Variable(bias)

		# save these for later
		self.poolSize = poolSz
		self.params = [self.W, self.b]

	def forward(self, X):
		"""
		:param X: tf.Variable representing data to push through
			the layer
		:return: output tensorflow expression for conv/pooling layer
		"""

		# set up a basic convolutional layer. Add the bias
		# strides is how many pixels the filter moves per
		# multiply
		conv_out = tf.nn.conv2d(X, self.W, strides=[1,1,1,1], padding='SAME')
		conv_out = tf.nn.bias_add(conv_out,self.b)

		# image dimensions are expected to b n*width*height*cho
		# n is the number of samples
		# width and height are the output image dimensions
		# cho is the number of output channels
		p1, p2 = self.poolSize
		pool_out = tf.nn.max_pool(
			conv_out,
			ksize=[1,p1,p2,1], #pool_size
			strides=[1,p1,p2,1],
			padding='SAME'
		)

		# return an activation function over all of it
		return tf.nn.relu(pool_out)

class ClassificationCNN(object):
	def __init__(self, shapein, convpoolSizes,denseSizes, k):
		"""
		:param shapein: tuple - dimensions of the input images
			(n, w, h, c)
		:param convpoolSizes: list - each entry is the size of a
			convultion/pooling layer to make
		:param denseSizes: list - each entry is the size of a
			fully-connected layer to make
		:param k: int - number of categories we are classifying
			these images into
		"""
		# set aside for later use
		print("getting Layer sizes")
		self.convpool_layer_sizes = convpoolSizes
		self.dense_layer_sizes = denseSizes

		# set aside for use in loop
		n, width, height, c = shapein
		outw, outh, chi = width, height, c


		self.widthIn = width
		self.heightIn= height
		self.channelsIn=c
		self.categoriesOut = k

		# list to hold convolutional layers
		print("making convpool layers")
		print(shapein)
		self.convpool_layers = []
		for cho, fw, fh in self.convpool_layer_sizes:
			layer = ConvPoolLayer(chi,cho,fw,fh)
			self.convpool_layers.append(layer)

			outw = outw // 2 #remember we also pool the layers
			outh = outh // 2 #so shrink them by a factor of 2

			#outw comes out as 56 when it should be 57. This is because of the integer division by 2.
			#for now I will hard code the remedy
			if outw == 56:
				outw+=1

			chi = cho # this layer's output in the next one's
				  # input

			print("(?,",outw, outh, chi, ")")




		#exit()

		# list to hold fully connected layers
		print("Making dense layers")
		self.dense_layers = []

		# initialize mlp layers
		# size of input must be the same as the output size
		# from the last convpool layer
		# self.convpool_layer_size[-1][0] gets the last CP layer's
		# number of channels. CP layers output 4-dimensional data.
		# Fully-connected layers input 2-dimensional data, So we
		# need to flatten the output from the last CP layer.
		# This code amounts to
		# channels out * output width * output height *
		ch1 = self.convpool_layer_sizes[-1][0]*outw*outh

		# count is to give the dense layers ids for some reason
		count = 0
		for ch2 in self.dense_layer_sizes:
			layer = DenseLayer(ch1,ch2,count)
			self.dense_layers.append(layer)
			ch1 = ch2
			count += 1

		# define the last dense layer to outpu (which is a
		# logistic layer).
		# Reminder that k is the number categories we are
		# classifying our images into.
		# ch1 is the number of nodes in the previous layer.
		# W will be a (ch1,k)-sized matrix while b will be a
		# k-sized vector
		print("making self layer")
		W, b =initWeightAndBias(ch1,k)
		self.W = tf.Variable(W, 'W_logreg')
		self.b = tf.Variable(b, 'b_logreg')

		print("collecting params")
		# collect parameters in one place for later use
		self.params = [self.W,self.b]
		for h in self.convpool_layers:
			self.params += h.params
		for h in self.dense_layers:
			self.params += h.params

	def forwardOp(self,X):
		"""
		:param X: tensorflow variable - data to push through cnn
		:return: tensorflow equation - tf object for use in 
			tf session
		"""

		# push data through the first half of the cnn
		Z=X
		for layer in self.convpool_layers:
			Z = layer.forward(Z)

		# shape of data in the previous layer is 4D the next layer
		# has 2D input so we need to flatten the data
		# Z_shape: (n, w, h, ch)
		# np.prod(z_shape[1:]) = outw*outh*ch
		Z_shape = Z.get_shape().as_list()
		print("Z_shape",Z_shape)
		#Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
		Z = tf.reshape(Z, [-1, Z_shape[1]*Z_shape[2]*Z_shape[3] ] )
		# Z.getShape() is now a (n,w*h*ch) or something like that

		for layer in self.dense_layers:
			Z = layer.forward(Z)

		# Finally push it through the self layer.
		# I'm not sure why the last layer does not have an
		# activation function
		return tf.matmul(Z,self.W) + self.b

	def predictOp(self, X):
		"""
		:param X: tf.Variable - data we want prediction(s) for
		:return: tf equation - represents the prediction process
		"""
		pY = self.forwardOp(X)
		return tf.argmax(pY,1)

	def fit(self, Generator, TestGenerator, lr=1e-3, mu=0.99, reg=1e-3, decay=0.99999, eps= 1e-2, epochs=20, show_fig=True):
		"""
		cnn fit operation
		:param Generator: yields an iteratable of training data
		:param TestGenerator: iterable of testing data
		:param lr: float - learning rate
		:param mu: float
		:param reg: float
		:param decay: float
		:param eps: float
		:param epochs: int - number of runs through the data
		:param show_fig: bool - do you want to display a graph with the results?
		"""
		# convert everything to np.float32
		lr = np.float32(lr)
		mu = np.float32(mu)
		reg = np.float32(reg)
		decay = np.float32(decay)
		eps = np.float32(eps)

		# set up tensorflow functions and variable
		# None here is used to indicate that we don't yer know how
		# many samples will be in each batch.
		tfX = tf.placeholder(tf.float32, shape=(None, self.widthIn,self.heightIn,self.channelsIn), name='X')
		tfY = tf.placeholder(tf.float32, shape=(None, self.categoriesOut), name='Y')

		# this will be our comp graph
		# I believe tf uses this to construct the gradient descent
		# equations that will tune our parameters. We will use
		# this to calculate the how far the model is from where we
		# want it to be, also known as the cost.
		act = self.forwardOp(tfX)

		# Total all weights and biases. I don't understant all the
		# math here. for each parameter compute th l2 loss of it.
		# Sum those, and multiply that by reg.
		# returns a lazy function for tf to use
		rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
		# some kind of cost functoin to tune this machine quickly
		cost = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(
				logits=act,
				labels=tfY
			)
		) + rcost


		prediction = self.predictOp(tfX)

		# thie will be the function to train the machine.
		# notice how it is set to minimize the cost function from
		# earlier.
		train_op = tf.train.RMSPropOptimizer(lr, decay=decay,momentum=mu).minimize(cost)


		# Here we will test the model with a single run


		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)

			Ybatch1, Xbatch1  = Generator.__next__()

			#get the first layer
			convpool_layer1 = self.convpool_layers[0]

			#create the convolutional layer
			conv1 = tf.nn.conv2d(tfX, convpool_layer1.W, strides=[1,1,1,1], padding='SAME')
			conv1_bias = tf.nn.bias_add(conv1,convpool_layer1.b)

			#create the pooling layer
			p1,p2 = convpool_layer1.poolSize
			pool_out1 = tf.nn.max_pool(
				conv1_bias,
				ksize=[1,p1,p2,1],
				strides=[1,p1,p2,1],
				padding='SAME'
			)

			conv1_finalout = tf.nn.relu(pool_out1)

			#test it
			conv1_shape = sess.run(tf.shape(conv1), feed_dict={tfX:Xbatch1})
			print("conv1_shape",conv1_shape)
			print()
			del conv1_shape

			conv1_final_shape = sess.run(tf.shape(conv1_finalout), feed_dict={tfX:Xbatch1})
			print("conv1_final_shape", conv1_final_shape)
			print()
			del conv1_final_shape

			shape_finaloutputs = sess.run(tf.shape(act),feed_dict={tfX:Xbatch1})
			print("shape of final outputs",shape_finaloutputs)
			print()
			del shape_finaloutputs
		#costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			i = 0
			while i<epochs:
				i+=1
				j = 0
				for Ybatch, Xbatch in generator:
					j+=1
					sess.run(train_op, feed_dict={tfX:Xbatch, tfY:Ybatch})
					if j%20 == 0:
						c = session.run(cost,feed_dict={tfx:Xbatch,tfY:Ybatch})
						print("cost:",c)
						del c


def main():
	print("main")
	bs = 8
	model = ClassificationCNN(
		shapein = (bs,450,600,3),
		convpoolSizes = [(20,11,11),(20,11,11),(20,5,5)],
		denseSizes = [300,100],
		k=7
		)
	batcher = bt.Batcher()
	model.fit(batcher.generate(bs),batcher.testGenerate(bs))

if __name__ == "__main__":
	main()
