import report_tools as rt
import cv2 as cv
import numpy as np
from sklearn.utils import shuffle


fs = rt.read_files_sanitized(2)

bundleDir = "bundles/"

def y2Indicator(n,s):
	return [1 if n==i else 0 for i in range(s) if True]

def bundle(size, filesList):
	bundle = []
	tbundle=[]
	count = 0
	for c,f in filesList:
		bundle.append(cv.imread(f))
		tbundle.append(y2Indicator(c,7))
		count+=1
		if count >= size:
			count=0
			pack = np.array(bundle)
			tpack= np.array(tbundle)
			yield tpack, pack
			del pack
			del tpack
			bundle=[]


def toFile(file,data):
	np.save(file,data)

def fromFile(file):
        return np.load(file)


fs = shuffle(fs)

i = 1
update = 50
for t,b in bundle(64, fs):
	bFile = bundleDir+str(i)+'.dat'
	tFile = bundleDir+str(i)+'.cat'

	fh = open(bFile,mode="w")
	toFile(fh,b)
	fh.close()

	fh = open(tFile,mode="w")
	toFile(fh,t)
	fh.close()

	fh = open(bFile,mode='r')
	c = fromFile(fh)
	fh.close()

	fh = open(tFile,mode='r')
	s = fromFile(fh)
	fh.close()

	if i%update==1:
		print(str(i)+' b: '+str(np.allclose(b,c))+' t:'+str(np.allclose(t,s)))

	#print(np.allclose(b,c))
	i+=1
