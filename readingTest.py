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
	numbundles = 0
	progess_check = 50
	list_of_errors = []
	for c,f in filesList:
		try:
			bundle.append(cv.imread(f))
			tbundle.append(y2Indicator(c,7))
			count+=1
			if count >= size:
				pack = np.array(bundle)
				tpack= np.array(tbundle)
				#yield tpack, pack
				del pack
				del tpack
				bundle=[]
				if numbundles%progess_check==0:
					print(".")
				count=0
				numbundles+=1
		except KeyboardInterrupt as e:
			print("Keyboard Interrupt")
			exit()
		except Exception as e:
			print(str(numbeundles)+" "+str(e)+" ")
			list_of_errors.append(e)
	return list_of_errors


print("running")
for e in bundle(64, fs):
	print(str(e))
