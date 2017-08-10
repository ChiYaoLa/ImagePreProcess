import numpy as np
from matplotlib import pyplot as plt

import shutil



def GetFileList(FindPath,FlagStr=[]):
	import os
	FileList=[]
	FileNames=os.listdir(FindPath)
	if len(FileNames)>0:
		for fn in FileNames:
			if len(FlagStr)>0:
				if IsSubString(FlagStr,fn):
					fullfilename=os.path.join(FindPath,fn)
					FileList.append(fullfilename)
			else:
				fullfilename=os.path.join(FindPath,fn)
				FileList.append(fullfilename)


	if len(FileList)>0:
		FileList.sort()

	return FileList
def IsSubString(SubStrList,Str):
	flag=True
	for substr in SubStrList:
		if not(substr in Str):
			flag=False

	return flag

txt=open('G:\caffe2\ding9QRrecognition\\train.txt','w')

imgfile=GetFileList('G:\caffe2\ding9QRrecognition\\trainData\D9train')
for img in imgfile:
	str=img+'\t'+'1'+'\n'
	txt.writelines(str)

imgfile=GetFileList('G:\caffe2\ding9QRrecognition\\trainData\QRtrain')
for img in imgfile:
	str=img+'\t'+'0'+'\n'
	txt.writelines(str)
txt.close()


txt=open('G:\caffe2\ding9QRrecognition\\test.txt','w')
imgfile=GetFileList('G:\caffe2\ding9QRrecognition\\testData\D9test')
for img in imgfile:
	str=img+'\t'+'1'+'\n'
	txt.writelines(str)


imgfile=GetFileList('G:\caffe2\ding9QRrecognition\\testData\QRtest')
for img in imgfile:
	str=img+'\t'+'0'+'\n'
	txt.writelines(str)
txt.close()

