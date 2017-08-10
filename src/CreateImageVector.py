# coding=utf-8
#用各种图像操作生成多个文件夹
#每个图像处理文件夹都生成一个list
#list中用随机数根据索引 从多个文件夹提取对应路径 赋值图片追加到新的目录下面
import os
import cv2
import math
import random
import shutil
from PIL import Image

def CreateList(dirpath):
    ImgList=[]
    ImgNames=os.listdir(dirpath)
    if len(ImgNames)!=0 :
        for i in ImgNames :
            FullPath=os.path.join(dirpath,i)
            ImgList.append(FullPath)

    print('CreateList Ok')

    return ImgList

#下面是一波炫酷的图像处理操作
#错切操作
def Warp(image,angle):
    a = math.tan(angle*math.pi/180.0)
    W = image.width
    H = int(image.height+W*a)
    size = (W,H)
    iWarp = cv2.cv.CreateImage(size,image.depth,image.nChannels)
    for i in range(image.height):
        for j in range(image.width):
            x = int(i+j*a)
            iWarp[x,j] = image[i,j]
    return iWarp

#旋转操作   这个api有bug 不能逆时针转动
def Rotate(angle,ImgPath):
    pil_im = Image.open(ImgPath)
    pil_im = pil_im.rotate(angle)

    return pil_im

#缩放操作  我觉得size不是一个好的输入参数  size是元祖类型
def ResizeImg(size,ImgPath):
    pil_im = Image.open(ImgPath)
    pil_im = pil_im.resize(size)

    return pil_im



#生成的List 将所有的图片 分别调用图像处理函数 再将生成的图片放到各自新的文件夹
# ImgList是orignal图片的图片列表 DirImgNum是对一张图片同一操作随意生成的图片次数  DirPath你要放的新建的目录
def CreateDiffImgDirs(ImgList,DirImgNums,DirPath):

    ListIndex = 0

    if (os.path.exists(DirPath) == False):
        os.makedirs(DirPath)

    for imgpath in ImgList:
        InputImg = cv2.cv.LoadImage(imgpath, 0)   #1是彩图 0是二值化
        for num in range(0,DirImgNums,1):

            #每张图都来一波不一样的图像操作

            #错切
            outImg = Warp(InputImg,random.randint(0,10))   #用随机数生成错切角度
            # NewFullPath=os.path.join(DirPath,num)      #这个num是路径么？一个问号
            path = os.path.join(DirPath,'cuoqie_'+str(ListIndex)+'_'+str(num)+'.jpg')
            cv2.cv.SaveImage(path,outImg)

            # #旋转
            # outImg = Rotate(20, InputImg)
            # path = os.path.join(DirPath,'rotate_'+str(ListIndex)+'_'+str(num)+'.jpg')
            # cv2.cv.SaveImage(path,outImg)

            # #resize操作
            # outImg = ResizeImg((random.randint(128,256),random.randint(128,256)),InputImg)
            # path = os.path.join(DirPath,'resize_'+str(ListIndex)+'_'+str(num)+'.jpg')
            # cv2.cv.SaveImage(path,outImg)

            print(str(ListIndex)+'-'+str(num)+'ok')
        ListIndex += 1

    return DirPath

#选
def SelectNetInputImages():
    OrignalPath = 'G:\caffe2\ding9QRrecognition\\trainData\D9train'         #自己定义
    DirImgNums=10         #可以定义
    SampleNums = 100    # 建议 DirImgNums*len(OrignalImgList) > SampleNums*2
    NetInputImgDir = 'G:\caffe2\ding9QRrecognition\GenerateInputImgDir'      #自己定义

    OrignalImgList = CreateList(OrignalPath)

    print (OrignalImgList)

    InputImageDir = CreateDiffImgDirs(OrignalImgList,DirImgNums,NetInputImgDir)

    NetInputImgList = CreateList(InputImageDir)

    SliceNetInputImg = random.sample(NetInputImgList,SampleNums)   #还是list 真是简洁

    print('SliceNetInputImg create ok！')
    print(SliceNetInputImg)

    return  SliceNetInputImg

#根据Imglist 将所有图片copy一份新的目录下面  这个新目录就是你的train imges或者 test imges所在的目录
def MakeNewImgDir(ImgList,NewImgDir):
    if len(ImgList)!= 0 :
        for oripath in ImgList:
             newpath = os.path.join(NewImgDir,os.path.basename(oripath))
             shutil.copyfile(oripath,newpath)

    print('copy imges generate ok! now you have great number of net input images!!')

    return True


#下面开始执行

SelectNetInputImagesList = SelectNetInputImages()
# print SelectNetInputImages()    #测试1 看看输出的list对不对  这相当于在执行了一遍  好傻

trainImageDir = 'G:\caffe2\ding9QRrecognition\\trainData\ceshi'  #自己定义

MakeNewImgDir(SelectNetInputImagesList,trainImageDir)
