from datetime import datetime as dt
import csv

#reportFile = open("report.txt", mode='w')
full_path = "/media/stefan/08D5-F97D/"

filesListName = "filesList.txt"
filesListBackupName = "filesListBackup.txt"
filesListSanitizedNameFormat = "filesListSanitized%d.txt"


#used to remove everything in the filesList file
def clobber_files_list():
	filesList = open(filesListName,mode='w')
	#filesList.write("")
	filesList.close()

##appends the given files to the filesList file
def report_files_made(searchTerm,list):
	filesList  = open(filesListName, mode='a')
	for file_name in list:
		filesList.write(searchTerm +', ' + file_name[len(full_path):]+'\n')
	filesList.close()

#moves the contents of the filesList File to the filesListBackupFile 
def backup_files_list():
	filesListBackup = open(filesListBackupName, mode="a")
	filesListBackup.write(str(dt.now())+"\n")
	filesList = open(filesListName, mode="r")
	filesListBackup.write(filesList.read())
	filesListBackup.write('\n')
	filesList.close()
	filesListBackup.close()

def report_files_sanitized(filter_num, files_list): 
	with open(filesListSanitizedNameFormat%(filter_num), mode="w") as filesListSanitized:
		csvwriter = csv.writer(filesListSanitized, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for item in files_list:
			csvwriter.writerow(item)

def read_files_sanitized(filter_num):
	files_list = []
	with open(filesListSanitizedNameFormat%(filter_num), mode='r') as filesListSanitized:
		csvreader = csv.reader(filesListSanitized, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for row in csvreader:
			files_list.append(row)
	return files_list
