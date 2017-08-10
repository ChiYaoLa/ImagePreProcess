# coding=utf-8
import os
import cv2
import math
import random
import shutil
import numpy as np

DemoImagePath='abc.jpg'  #Demo图片绝对或者相对路径
##图像旋转参数
RotateOrign=(120,120) #表示旋转中心，若rows,cols=img.shape[:2]，那么(rows/2,clos/2)就是图片中心点
RotateAngle=80    #表示顺时针旋转80度
RotateScale=0.8  #表示图像旋转之后会缩放到原来的0.8
##图像平移参数
MoveX=60  #分别表示向X或Y方向随机移动多少像素单位。负数表示反向 范围上下限如左
MoveY=50
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
AdaptiveTtype=1   #1表示阈值取自相邻区域平均值 2表示阈值相邻区域加权和且权重是高斯窗口
AdaptiveTBlocksize=5 #邻域大小规定必须是奇数且大于1
AdaptiveTBias=2   #可调的常数范围，等于平均值或者加权平均值减去这个常数
##卷积操作
CKernelMatrix=np.float32([[1,2,3],[2,3,4],[2,5,1]])/10  #注意了要用float，有些更简洁 np.ones((5,5),np.float32)/25
CSuiJiBias=(-1,1)     #偏置的范围
##高斯模糊
GaussianBlurkernelSize=5  #高斯卷积核的尺寸 必须是奇数 大于1
GaussianBlurSigma=0      #标准差范围
#梯度滤波 高通滤波器Sobel Scharr和Lapcian 都是图像微分操作
GradientFilterType=1           #1使用Sobel 2使用Lapcian  随机生成,如果你只要1或者2 就写(1,1) 或(2,2)
GradientFilterSobelDirection=1  #面向Sobel算子的，1表示x和y两个方向 2表示只在x方向 3表示只在y方向 随机生成,
                                     # 接上面，如果只要某一个方向就写(1,1) (2,2),(3,3) 或者某两个如(2,3)
GradientFilterKernelSize=5      #卷积核大小  必须是奇数 不大于31 大于1
##边缘检测
EdgeDetectionThreshold1=25    #阈值下限
EdgeDetectionThreshold2=240   #阈值上限
EdgeDetectionBias=(10,20)          #偏置

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
def nothing():
    pass

cv2.namedWindow('panel',cv2.WINDOW_NORMAL)
# cv2.createTrackbar('RotateOrign','panel',0,500,nothing)
# cv2.createTrackbar('RotateAngle','panel',0,360,nothing)
# cv2.createTrackbar('RotateScale','panel',0,2,nothing)
cv2.createTrackbar('1_MoveX','panel',-250,250,nothing)
cv2.createTrackbar('1_MoveY','panel',-250,250,nothing)
# cv2.createTrackbar('ATSuiJiBias','panel',-20,20,nothing)
# cv2.createTrackbar('PTSuiJiBias','panel',-50,50,nothing)
cv2.createTrackbar('3_ATtype','panel',1,2,nothing)
cv2.createTrackbar('3_ATBias','panel',-5,5,nothing)
# cv2.createTrackbar('CSuiJiBias','panel',-5,5,nothing)
cv2.createTrackbar('5_Sigma','panel',0,5,nothing)
cv2.createTrackbar('6_FilterType','panel',1,2,nothing)
cv2.createTrackbar('6_SobelDirection','panel',1,2,nothing)
cv2.createTrackbar('7_Threshold1','panel',0,250,nothing)
cv2.createTrackbar('7_Threshold2','panel',250,500,nothing)
cv2.createTrackbar('7_Bias','panel',0,50,nothing)




while 1:
    # RotateOrign=cv2.getTrackbarPos('RotateOrign','panel')
    # RotateAngle=cv2.getTrackbarPos('RotateAngle','panel')
    # RotateScale=cv2.getTrackbarPos('RotateScale','panel')
    MoveX=cv2.getTrackbarPos('1_MoveX','panel')
    MoveY=cv2.getTrackbarPos('1_MoveY','panel')
    # ATSuiJiBias=cv2.getTrackbarPos('ATSuiJiBias','panel')
    # PTSuiJiBias=cv2.getTrackbarPos('PTSuiJiBias','panel')
    AdaptiveTtype=cv2.getTrackbarPos('3_ATtype','panel')
    AdaptiveTBias=cv2.getTrackbarPos('3_ATBias','panel')
    # CSuiJiBias=cv2.getTrackbarPos('CSuiJiBias','panel')
    GaussianBlurSigma=cv2.getTrackbarPos('5_Sigma','panel')
    GradientFilterType=cv2.getTrackbarPos('6_FilterType','panel')
    GradientFilterSobelDirection=cv2.getTrackbarPos('6_SobelDirection','panel')
    EdgeDetectionThreshold1=cv2.getTrackbarPos('7_Threshold1','panel')
    EdgeDetectionThreshold2=cv2.getTrackbarPos('7_Threshold2','panel')
    EdgeDetectionBias=cv2.getTrackbarPos('7_Bias','panel')

    # RotateOutImg=Rotate(DemoImagePath,RotateOrign,RotateAngle,RotateScale)
    TranslateOutImg=Translate(DemoImagePath,MoveX,MoveY)
    # outImg=AffineTransformation(DemoImagePath,ATPreMatrix,ATNextMatrix)   感觉这个函数要重构下
    OtsuBinarizationOutImg=OtsuBinarization(DemoImagePath)
    AdaptiveThresholdOutImg=AdaptiveThreshold(DemoImagePath,AdaptiveTtype,AdaptiveTBlocksize,AdaptiveTBias)
    ConvolutionOutImg=Convolution(DemoImagePath,CKernelMatrix)
    GaussianBlurOutImg=GaussianBlur(DemoImagePath,GaussianBlurkernelSize,GaussianBlurSigma)
    GradientFilterOutImg=GradientFilter(DemoImagePath,GradientFilterType,GradientFilterSobelDirection,GradientFilterKernelSize)
    EdgeDetectionOutImg=EdgeDetection(DemoImagePath,EdgeDetectionThreshold1,EdgeDetectionThreshold2)

    # cv2.imshow('RotateOutImg',RotateOutImg)
    cv2.imshow('1_TranslateOutImg',TranslateOutImg)
    cv2.imshow('2_OtsuBinarizationOutImg',OtsuBinarizationOutImg)
    cv2.imshow('3_AdaptiveThresholdOutImg',AdaptiveThresholdOutImg)
    cv2.imshow('4_ConvolutionOutImg',ConvolutionOutImg)
    cv2.imshow('5_GaussianBlurOutImg',GaussianBlurOutImg)
    cv2.imshow('6_GradientFilterOutImg',GradientFilterOutImg)
    cv2.imshow('7_EdgeDetectionOutImg',EdgeDetectionOutImg)

    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()  #还不能写在while里面 按一个键就全没有了 为啥？
