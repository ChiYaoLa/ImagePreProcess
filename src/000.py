# coding=utf-8
import os
import cv2
import math
import random
import shutil
import numpy as np

#独立出来的参数接口，用户根据自己的需要来赋值
##全局参数
OrignalImageDirPath='G:\caffe2\ding9QRrecognition\PyWork\src\OrignalImageDirPath'   #原始图片文件夹的相对或者绝对路径 （github不能上传太多图片。。所以我就没有这个文件夹）
ImgLibDirPath='G:\caffe2\ding9QRrecognition\PyWork\src\ImgLibDirPath'        #新生成的图片所在的文件夹（量很大，所以叫库）
TotalImgNumInLib=50000      #表示你希望生成1万张，原始图片数*图像操作数*每种操作随机生成图片数=最后生成图片总数，所以取值不要太偏激
ImageType='.jpg'            #原始图片以及处理完之后图片的保存格式
SelectSampleMode=1  #目前我也就只开发了 随机抽样 以后再补充吧
SelectSampleNum=10      #抽取多少图片 别比TotalImgNumInLib大
SelectSampleImgDir='G:\caffe2\ding9QRrecognition\PyWork\src\SelectSampleImgDir'    #抽样后图片要放的位置

##是否使用某个图像操作 都对应下面每一个操作 1表示使用 0表示禁用
UseMethodOrNot={
"Use_Rotate_OrNot":1,
"Use_Translate_OrNot":1,
"Use_AffineTransformation_OrNot":1,
"Use_PerspectiveTransformation_OrNot":1,
"Use_OtsuBinarization_OrNot":1,
"Use_AdaptiveThreshold_OrNot":1,
"Use_Convolution_OrNot":1,
"Use_GaussianBlur_OrNot":1,
"Use_GradientFilter_OrNot":1,
"Use_EdgeDetection_OrNot":1  }
# ##图像错切参数
# WarpLowAngle=0 #表示随机生成在[WarpLowAngle,WarpUpAngle)之间的角度
# WarpUpAngle=10
##图像旋转参数
RotateOrign=(250,250) #表示旋转中心，若rows,cols=img.shape[:2]，那么(rows/2,clos/2)就是图片中心点
RotateAngle=(80,90)  #表示顺时针旋转80到90度随机
RotateScale=(0.8,1)  #表示图像旋转之后会缩放到原来的0.8到1倍 随机
##图像平移参数
MoveX=(60,100)  #分别表示向X或Y方向随机移动多少像素单位。负数表示反向 范围上下限如左
MoveY=(-50,50)
##仿射变换数学关系  http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=warpaffine
ATPreMatrix=[[50,50],[200,50],[50,200]]   #注意都必须是3*2的矩阵 不懂就去看上面的链接
ATNextMatrix=[[10,100],[100,50],[100,250]]
ATSuiJiBias=(0,5)                  #在维数不变，我给上面的映射每个点加上随机偏置量ATSuiJiBias
##透视变化 去opencv去查吧 链接太长了
PTPreMatrix = [[100,100],[368,52],[28,237],[389,390]]    #这里是透视前后区域的四个点坐标组成的list
PTNextMatrix = [[100,100],[100,400],[400,400],[100,400]]
PTArea = (300,300)                                       #透视之后区域的大小 与上一行保持数学相等
PTSuiJiBias=(-5,5)         #在上面数学关系依然成立的前提下，整体加上随机量PTSuiJiBias，范围如左
##自适应阈值二值化
AdaptiveTtype=(1,2)   #1表示阈值取自相邻区域平均值 2表示阈值相邻区域加权和且权重是高斯窗口
AdaptiveTBlocksize=5 #邻域大小规定必须是奇数且大于1
AdaptiveTBias=(2,5)   #可调的常数范围，等于平均值或者加权平均值减去这个常数
##卷积操作
CKernelMatrix=np.float32([[1,2,3],[2,3,4],[2,5,1]])/10  #注意了要用float，有些更简洁 np.ones((5,5),np.float32)/25
CSuiJiBias=(-1,1)     #偏置的范围
##高斯模糊
GaussianBlurkernelSize=5  #高斯卷积核的尺寸 必须是奇数 大于1
GaussianBlurSigma=(0,2)      #标准差范围
#梯度滤波 高通滤波器Sobel Scharr和Lapcian 都是图像微分操作
GradientFilterType=(1,2)            #1使用Sobel 2使用Lapcian  随机生成,如果你只要1或者2 就写(1,1) 或(2,2)
GradientFilterSobelDirection=(1,2)  #面向Sobel算子的，1表示x和y两个方向 2表示只在x方向 3表示只在y方向 随机生成,
                                     # 接上面，如果只要某一个方向就写(1,1) (2,2),(3,3) 或者某两个如(2,3)
GradientFilterKernelSize=5      #卷积核大小  必须是奇数 不大于31 大于1
##边缘检测
EdgeDetectionThreshold1=25    #阈值下限
EdgeDetectionThreshold2=240   #阈值上限
EdgeDetectionBias=(10,20)          #偏置

# # 一波图像处理操作模块
# ## 图像错切
# def Warp(image,angle):
#     a = math.tan(angle*math.pi/180.0)
#     W = image.width
#     H = int(image.height+W*a)
#     size = (W,H)
#     iWarp = cv2.cv.CreateImage(size,image.depth,image.nChannels)
#     for i in range(image.height):
#         for j in range(image.width):
#             x = int(i+j*a)
#             iWarp[x,j] = image[i,j]
#     return iWarp

## 旋转操作
def Rotate(image,rotateOrign,rotateAngle,rotateScale):
    img=cv2.imread(image)
    rows,cols=img.shape[:2]
    M=cv2.getRotationMatrix2D(rotateOrign,rotateAngle,rotateScale)
    dst=cv2.warpAffine(img,M,(rows,cols))
    return dst

## 平移操作
def Translate(image,moveX,moveY):
    img=cv2.imread(image)
    M=np.float32([[1,0,moveX],[0,1,moveY]])
    dst=cv2.warpAffine(img,M,img.shape[:2])
    return dst

##仿射变换
def AffineTransformation(image,ATpreMatrix,ATnextMatrix):
    img=cv2.imread(image)
    rows,cols=img.shape[:2]
    psn1=np.float32(ATpreMatrix)
    psn2=np.float32(ATnextMatrix)
    M=cv2.getAffineTransform(psn1,psn2)
    dst=cv2.warpAffine(img,M,(rows,cols))
    return dst

##透视变换
def PerspectiveTransformation(image,PTpreMatrix,PTnextMatrix,PTarea):
    img=cv2.imread(image)
    rows,cols=img.shape[:2]
    psn1=np.float32(PTpreMatrix)
    psn2=np.float32(PTnextMatrix)
    M=cv2.getPerspectiveTransform(psn1,psn2)
    dst=cv2.warpPerspective(img,M,PTarea)
    return dst

##otsu's  二值化
def OtsuBinarization(image):
    img=cv2.imread(image,0)
    ret,th=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

#自适应阈值 二值化
def AdaptiveThreshold(image,type,blocksize,bias):
    img=cv2.imread(image,0)
    if type==1:
        mode = cv2.ADAPTIVE_THRESH_MEAN_C
    if type == 2:
        mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    th=cv2.adaptiveThreshold(img,255,mode,cv2.THRESH_BINARY,blocksize,bias)
    return th

#卷积操作
def Convolution(image,kernelMatrix):
    img=cv2.imread(image)
    kernel=kernelMatrix
    dst=cv2.filter2D(img,-1,kernel)
    return dst

#高斯模糊 一种出名的卷积
def GaussianBlur(image,kernelSize,sigma):
    img=cv2.imread(image)
    Gblur=cv2.GaussianBlur(img,(kernelSize,kernelSize),sigma)
    return Gblur

#梯度滤波
def GradientFilter(image,type,sobelDirection=1,kernelSize=3):
    img=cv2.imread(image)
    if type==1:
        if sobelDirection==1:
            dst=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=kernelSize)
        if sobelDirection==2:
            dst=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernelSize)
        if sobelDirection==3:
            dst=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernelSize)
    elif type==2:
        dst=cv2.Laplacian(img,cv2.CV_64F)
    return dst

##边缘检测
def EdgeDetection(image,threshold1,threshold2,):
    img=cv2.imread(image)
    edges=cv2.Canny(img,threshold1,threshold2)
    return edges


#一波文件操作
##根据文件夹目录 创建图片路径List
def CreateList(dirpath):
    ImgList=[]
    ImgNames=os.listdir(dirpath)
    if len(ImgNames)!=0 :
        for i in ImgNames:
            FullPath=os.path.join(dirpath,i)
            ImgList.append(FullPath)
    print('CreateList Ok')
    return ImgList

#调用开启的图像操作，随机赋予参数，生成指定数量的图片并放到你指定的文件目录下
def GenerateImgLib(imgList,imgLibDirPath):
    listIndex=0
    imageType= ImageType
    totalOrignImgNums=len(imgList)
    print('原始图片总数是'+str(totalOrignImgNums))
    totalMethodNumsInUse=0      #你开启了多少种图像处理操作
    allMethodNowInUse=[]
    for key,value in UseMethodOrNot.items():
        if value==1:
            totalMethodNumsInUse += 1
            methodInUse=key.split('_')[1]
            allMethodNowInUse.append(methodInUse)    #将所有正在使用的图像操作加入list
    print('正在使用的方法数：'+str(totalMethodNumsInUse))
    print('totalMethodNumsInUse 不是0就行！现在是'+str(totalMethodNumsInUse))
    totalImgNumInLib =  TotalImgNumInLib

    generateImgNumPerMethod=round(float(totalImgNumInLib)/float(totalOrignImgNums)/float(totalMethodNumsInUse))
    if os.path.exists(imgLibDirPath)==True:
        shutil.rmtree(imgLibDirPath)
    if os.path.exists(imgLibDirPath)==False:
        os.makedirs(imgLibDirPath)

    for item in imgList:
        for num in range(0,generateImgNumPerMethod,1):
            # if 'Warp' in allMethodNowInUse:
            #     inputImg = cv2.cv.LoadImage(item, 0)                               #1是彩图 0是二值化
            #     outImg = Warp(inputImg,random.uniform(WarpLowAngle,WarpUpAngle))   #用随机数生成错切角度
            #     path = os.path.join(imgLibDirPath,'warp_'+str(listIndex)+'_'+str(num)+imageType)
            #     cv2.cv.SaveImage(path,outImg)
            if 'Rotate' in allMethodNowInUse:
                RotateAngleTmp=random.uniform(RotateAngle[0],RotateAngle[1])
                RotateScaleTmp=random.uniform(RotateScale[0],RotateScale[1])
                outImg=Rotate(item,RotateOrign,RotateAngleTmp,RotateScaleTmp)
                path = os.path.join(imgLibDirPath,'Rotate_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'Translate' in allMethodNowInUse:
                outImg=Translate(item,random.randint(MoveX[0],MoveX[1]),random.randint(MoveY[0],MoveY[1]))
                path = os.path.join(imgLibDirPath,'Translate_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'AffineTransformation' in allMethodNowInUse:
                for y in range(0,len(ATPreMatrix[0])):
                    for x in range(0,len(ATPreMatrix)):
                        ATPreMatrix[x][y]=ATPreMatrix[x][y]+random.randint(ATSuiJiBias[0],ATSuiJiBias[1])
                        ATNextMatrix[x][y]=ATNextMatrix[x][y]+random.randint(ATSuiJiBias[0],ATSuiJiBias[1])
                outImg=AffineTransformation(item,ATPreMatrix,ATNextMatrix)
                path = os.path.join(imgLibDirPath,'AffineTransformation_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'PerspectiveTransformation' in allMethodNowInUse:
                PTSuiJiBiasTmp=random.randint(PTSuiJiBias[0],PTSuiJiBias[1])
                for y in range(0,len(PTPreMatrix[0])):
                    for x in range(0,len(PTPreMatrix)):
                        PTPreMatrix[x][y]=PTPreMatrix[x][y]+PTSuiJiBiasTmp
                        PTNextMatrix[x][y]=PTNextMatrix[x][y]+PTSuiJiBiasTmp
                outImg=PerspectiveTransformation(item,PTPreMatrix,PTNextMatrix,PTArea)
                path = os.path.join(imgLibDirPath,'PerspectiveTransformation_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'OtsuBinarization' in allMethodNowInUse:
                outImg=OtsuBinarization(item)
                path = os.path.join(imgLibDirPath,'OtsuBinarization_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'AdaptiveThreshold' in allMethodNowInUse:
                AdaptiveTtypeTmp=random.randint(1,2)
                # AdaptiveTBlocksizeTmp=random.randint(AdaptiveTBlocksize[0],AdaptiveTBlocksize[1])
                AdaptiveTBiasTmp=random.randint(AdaptiveTBias[0],AdaptiveTBias[1])
                outImg=AdaptiveThreshold(item,AdaptiveTtypeTmp,AdaptiveTBlocksize,AdaptiveTBiasTmp)
                path = os.path.join(imgLibDirPath,'AdaptiveThreshold_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'Convolution' in allMethodNowInUse:
                for x in range(len(CKernelMatrix[0])):
                    for y in range(len(CKernelMatrix)):
                        CKernelMatrix[x][y]=CKernelMatrix[x][y]+random.uniform(CSuiJiBias[0],CSuiJiBias[1])
                outImg=Convolution(item,CKernelMatrix)
                path = os.path.join(imgLibDirPath,'Convolution_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'GaussianBlur' in allMethodNowInUse:
                # GaussianBlurkernelSizeTmp=random.randint(GaussianBlurkernelSize[0],GaussianBlurkernelSize[1])
                GaussianBlurSigmaTmp=random.randint(GaussianBlurSigma[0],GaussianBlurSigma[1])
                outImg=GaussianBlur(item,GaussianBlurkernelSize,GaussianBlurSigmaTmp)
                path = os.path.join(imgLibDirPath,'GaussianBlur_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'GradientFilter' in allMethodNowInUse:
                GradientFilterTypeTmp=random.randint(GradientFilterType[0],GradientFilterType[1])
                GradientFilterSobelDirectionTmp=random.randint(GradientFilterSobelDirection[0],GradientFilterSobelDirection[1])
                # GradientFilterKernelSizeTmp=random.randint(GradientFilterKernelSize[0],GradientFilterKernelSize[1])
                outImg=GradientFilter(item,GradientFilterTypeTmp,GradientFilterSobelDirectionTmp,GradientFilterKernelSize)
                path = os.path.join(imgLibDirPath,'GradientFilter_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)
            if 'EdgeDetection' in allMethodNowInUse:
                EdgeDetectionBiasTmp=random.randint(EdgeDetectionBias[0],EdgeDetectionBias[1])
                outImg=EdgeDetection(item,EdgeDetectionThreshold1,EdgeDetectionThreshold2)
                path = os.path.join(imgLibDirPath,'EdgeDetection_'+str(listIndex)+'_'+str(num)+imageType)
                cv2.imwrite(path,outImg)

        listIndex += 1

    print('当前listIndex是'+str(listIndex))


#下面这个函数 是一个补充而已，在你已经生成指定数量图片基础上，还可以根据你的需要抽取其中一部分合格的图片
#比如按比例抽样，我一时没有这个需求，觉得随机抽样就够了
#imgLibDirPath 是申城图片库所在位置  ，sampleImgDir是你抽样图片库要放的位置
def SelectSample(imgLibDirPath,mode,num,sampleImgDir):
    imgLibList=CreateList(imgLibDirPath)
    SelectList = random.sample(imgLibList,num)
    if os.path.exists(sampleImgDir)==True:
        shutil.rmtree(sampleImgDir)
    if os.path.exists(sampleImgDir)==False:
        os.makedirs(sampleImgDir)
        print('sampleImgDirchuangjiangchengg ')
    if len(SelectList) != 0:
        for item in SelectList:
            newpath=os.path.join(sampleImgDir,os.path.basename(item))
            shutil.copyfile(item,newpath)




#下面是自动执行的  你只需要把最前头的参数全都设定好就行
imgList = CreateList(OrignalImageDirPath)
GenerateImgLib(imgList,ImgLibDirPath)
SelectSample(ImgLibDirPath,SelectSampleMode,SelectSampleNum,SelectSampleImgDir)

