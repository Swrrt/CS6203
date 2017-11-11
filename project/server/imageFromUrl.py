import urllib2
import sys
import shutil
import os

def downloadImage(urlFile, downloadPath):
	if downloadPath and not downloadPath.endswith('/'): downloadPath += '/'
	f = open(urlFile,'r')
	i = 0
	try:
		os.mkdir(downloadPath)
	except:
		print("waring, path exists")

 	while True:
		fileName = f.readline()
		if not fileName: break
		if fileName <> '\n' :
			i = i + 1
			print("Downloading {}-th image, from {}, to {}{}.jpg".format(i,fileName,downloadPath,i))
			try:
				request = urllib2.urlopen("{}".format(fileName),timeout = 5)
				with open("{}{}.jpg".format(downloadPath,i),"w") as ff:
					ff.write(request.read())
				#urllib.urlretrieve("{}".format(fileName), "{}{}.jpg".format(downloadPath,i))
			except:
				print('fail to download')	
#			shutil.move("{}.jpg".format(i),"{}{}.jpg".format(downloadPath,i))
	f.close()
#fileList = "urlFile.txt"
#downloadImage(fileList,"/home/ubuntu/project/")
