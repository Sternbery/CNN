import report_tools as rt
import numpy as np
import cv2 as cv
import imp
from sklearn.utils import shuffle

def y2indicator(n,s):
	return [1 if n==i else 0 for i in range(s) if True]

class Batcher(object):
	def __init__(self):
		self.train = rt.read_files_sanitized(2)
		self.test = []

	def shuffle(self):
		self.train = shuffle(self.train)

	def setAsideTest(self,amount):
		self.train.append(self.test)
		self.test = self.test[-amount:]
		self.train= self.train[:-amount]

	def generate(self, batchSize):
		self.shuffle()
		bundle = []
		tbundle= []
		count  = 0
		for c,f in self.train:
			bundle.append(cv.imread(f))
			tbundle.append(y2indicator(c,7))
			count+=1

			if count >= batchSize:
				count = 0
				pack = np.array(bundle)
				tpack = np.array(tbundle)

				pack = pack/255.0
				yield tpack, pack
				bundle=[]

	def testGenerate(self, batchSize):
		bundle = []
		tbundle= []
		count  = 0
		for c,f in self.test:
			bundle.append(cv.imread(f))
			tbundle.append(y2indicator(c,7))
			count+=1

			if count == batchsize:
				count = 0
				pack = np.array(bundle)
				tpack= np.array(bundle)

				pack = pack / 255.0
				yield tpack, pack
				bundle = []

