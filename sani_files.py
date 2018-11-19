import cv2
import report_tools as rt
import os
import numpy as np

testimg = cv2.imread("downloads/Mammal/1. 188901-004-54df827e.jpg")
print(np.shape(testimg))
#exit()

def categoryMaker(s):
	cats = ["Mammal","Bird","Reptile","Fish","Amphibian","Bug","Invertebrate"]
	for i in range(len(cats)):
		if cats[i] in s:
			return i
	raise ValueError

correctShape = (450,600,3)

checkCount = 0
badCount = 0
def checkFileValid(filePath):
	global checkCount, badCount
	checkCount += 1
	try:
		#print(filePath)
		image = cv2.imread(filePath)
		if np.shape(image) != correctShape:
			print("Incorrect Shape: "+str(checkCount)+", "+str(np.shape(image))+" removed")
			badCount += 1
			return False
	except FileNotFoundError:
		print("FileNotFoundError: "+str(checkCount)+": removed")
		badCount += 1
		return False
	except ValueError:
		print ("ValueError: "+str(checkCount)+": removed")
		badCount += 1
		return False
	else:
		return True


rootDir = "downloads"
dirs = os.listdir(rootDir)

fileList = []


for d in dirs:
	imgNames = os.listdir(os.path.join(rootDir,d))
	for img in imgNames:
		imgPath= os.path.join(rootDir,d,img)
		#print(imgPath)
		if checkFileValid(imgPath):
			try:
				fileList.append((categoryMaker(d),imgPath))
			except ValueError:
				continue


print('bad: '+str(badCount)+' good: '+str(len(fileList)))

print(fileList)
rt.report_files_sanitized(2,fileList)

print('all done')
#print(type(test))
