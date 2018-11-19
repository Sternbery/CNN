import cv2 as cv2
from google_images_download import google_images_download as gid
import csv
import sys
import os
import report_tools as rt

#print(str(os.getcwd())+"/chromedriver")


#getting runtime motives
demoVal = 0
restartCat = None
restartAnim= None
restartOff = 0
clobbermotivepresent = False
plantoclobber = True
plantobackup = False
limit = 1000


#read command line directives
i=1
while i in range(len(sys.argv)):
	if sys.argv[i] == "demo1":
		demoVal = 1
		i+=1
		continue
	if sys.argv[i] == "demo2":
		demoVal = 2
		i+=1
		continue
	if sys.argv[i] == "test1":
		demoVal = 3
		i+=1
		continue
	if sys.argv[i] == "restart":
		mo = sys.argv[i+1]
		
		restartCat, restartAnim, restartOff = (mo.split(',') + [None, None])[:3]

		if restartAnim == "":
			restartAnim = None

		try:
			restartOff = int(restartOff)
		except:
			restartOff = None
		
		plantoclobber = clobbermotivepresent
		
		i+=2
		continue
	if sys.argv[i] == "limit":
		limit = int(sys.argv[i+1])
		i+=2
		continue
	if sys.argv[i] == "clobber":
		plantoclobber = True
		clobbermotivepresent = True
		i+=1
		continue
	if sys.argv[i] == "backup":
		plantobackup = True
		i+=1
		continue


#get data for CNN via Google Image Search
#constraints on search
arguments = {
	"keywords":"", #search term provided later
	"exact_size":"600,450", #we only wnat 600*450 sized images
	"format":"jpg", #we only want jpeg images
	#"output_directory":"/media/stefan/08D5-F97D", #where to store the images
	#"image_directory":"images", 
	#"no_numbering":"no_numbering", #do we number the images?
	"limit":str(limit),	#How many to download for each term
	"chromedriver": "/home/stefan/Downloads/chromedriver" #necessary software
	}

#this is where we'll store our search terms
keywords = {}



#demo breakpoint
if demoVal == 2:
	arguments["limit"] = "1"



#class.csv provided by kaggle. User MCI Machine Learning with additional data provided by me

#opens the class.csv file and puts the contents into a dictionary
with open('class.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        #split the animal names into a list and put them in the dictionary under their respective classification
        keywords[row["Class_Type"]] = row["Animal_Names"].split(', ')  
        line_count += 1
    print(f'Processed {line_count} categories.')
    csv_file.close()

#demo breakpoint
if demoVal == 1:
	print(keywords)
	sum = 0
	for category in keywords:
		sum += 1
		sum += len(keywords[category])
	print('total searches: '+ str(sum) )
	exit()


response = gid.googleimagesdownload()


# in case I have to pause the downloads
doneRestartingCat = True
doneRestartingAnim= True

if restartCat != "":
	doneRestartingCat = False
	if restartAnim != "":
		doneRestartingAnim = False

#demo breakpoint
if demoVal == 3:
	arguments["keywords"] = "Mammal"
	arguments["limit"] = "4"
	absolute_image_paths = response.download(arguments)
	print(absolute_image_paths)
	exit()

#for restarting an interrupted run
if restartCat != None:

	#this is equivalent
	#keywords = keywords.dropwhile(lambda cat: cat != restartCat)
	list_to_remove = []
	for category in keywords.keys():
		if category != restartCat:
			list_to_remove.append(category)
		else:
			break

	for category in list_to_remove:
		keywords.pop(category, None)


	if restartAnim != None:
		keywords[restartCat] = keywords[restartCat][ keywords[restartCat].index(restartAnim):]

if plantobackup:
	rt.backup_files_list()

if plantoclobber:
	rt.clobber_files_list()


#loop through each category
#do a google search for each keyword in that category and for the category itself
for category in keywords:
	
	if restartCat == None:
		searchTerm = category
		arguments["keywords"]=searchTerm
		absolute_image_paths = response.download(arguments)
		rt.report_files_made(searchTerm, absolute_image_paths[searchTerm])
	restartCat = None

	for name in keywords[category]:
		
		if restartOff != 0:
			arguments["offset"] = restartOff
			restartOff = 0

		searchTerm = f'{name} ({category})'
		arguments["keywords"]= searchTerm
		absolute_image_paths = response.download(arguments)

		#report on the files downloaded so we can access them later
		rt.report_files_made(searchTerm, absolute_image_paths[searchTerm])

		if "offset" in arguments:
			arguments.pop("offset", None)
	
	

print("all set")
