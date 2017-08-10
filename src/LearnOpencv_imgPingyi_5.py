# coding=utf-8
import cv2
import numpy as np


# #生成hsv 和mask
# cap = cv2.VideoCapture(0)
#
# ret,frame=cap.read()
# hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#
# lower_blue=np.array([110,50,50])
# upper_blue=np.array([130,255,255])
#
# mask = cv2.inRange(hsv,lowerb=lower_blue,upperb=upper_blue)
#
# cv2.imshow('frame',frame)
# cv2.imshow('mask',mask)
# cv2.imshow('hsv',hsv)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# #平移图片  OK
# img=cv2.imread('abc.jpg')
# M=np.float32([[1,0,-100],[0,1,-50]])
# dst=cv2.warpAffine(img,M,img.shape[:2])
# cv2.imshow('dst',dst)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #旋转操作 OK
# img=cv2.imread('abc.jpg')
# rows,cols=img.shape[:2]
# M=cv2.getRotationMatrix2D((rows/2,cols/3),67,0.8)
# dst=cv2.warpAffine(img,M,(rows,cols))
# cv2.imshow('dst',dst)
# cv2.imwrite('xuanzhuan.jpg',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #仿射变换 OK
# img=cv2.imread('abc.jpg')
# rows,cols=img.shape[:2]
#
# psn1=np.float32([[50,50],[200,50],[50,200]])
# psn2=np.float32([[10,100],[100,50],[100,250]])
#
# M=cv2.getAffineTransform(psn1,psn2)
# dst=cv2.warpAffine(img,M,(rows,cols))
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 透视变换
# img=cv2.imread('abc.jpg')
# rows,cols=img.shape[:2]
# psn1=np.float32([[100,100],[368,52],[28,237],[389,390]])
# psn2=np.float32([[100,100],[100,400],[400,400],[100,400]])
# M=cv2.getPerspectiveTransform(psn1,psn2)
# dst=cv2.warpPerspective(img,M,(300,300))
# cv2.imshow('img',img)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #图像阈值
# img=cv2.imread('abc.jpg')
# ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# cv2.imshow('img',thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #更厉害的阈值变换
# img=cv2.imread('abc.jpg',0)  #有0这个参数？？
# #中值滤波
# img=cv2.medianBlur(img,5)
# ret,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# #阈值取自相邻区域的平均值
# th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# #阈值取值取自相邻区域的加权和
# th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# cv2.imshow('th1',th1)
# cv2.imshow('th2',th2)
# cv2.imshow('th3',th3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #otsu's  二值化  略屌
# img=cv2.imread('abc.jpg',0)
# ret1,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# blur=cv2.GaussianBlur(img,(5,5),0)
# ret3,th3=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('th1',th1)
# cv2.imshow('th2',th2)
# cv2.imshow('th3',th3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #卷积 自定义卷积核 注意矩阵每个数字都是np.float32
# img=cv2.imread('abc.jpg')
# kernel=np.float32([[1,2,3],[2,3,4],[2,5,1]])/10
# # kernel=np.ones((5,5),np.float32)/25
# # print np.ones((5,5),np.float32)/25
# dst=cv2.filter2D(img,-1,kernel)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #卷积 高斯模糊
# Gblur=cv2.GaussianBlur(img,(5,5),10)
# cv2.imshow('Gblur',Gblur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #图像梯度
# img=cv2.imread('abc.jpg')
# laplacian=cv2.Laplacian(img,cv2.CV_64F)
# sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
# cv2.imshow('lapcian',laplacian)
# cv2.imshow('sobelx',sobelx)
# cv2.imshow('sobely',sobely)
# cv2.imshow('sobelxy',sobelxy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#边缘检测  带边缘检测条
def nothing():
    pass
cv2.namedWindow('image')
cv2.createTrackbar('yuzhi1','image',0,255,nothing)
cv2.createTrackbar('yuzhi2','image',0,255,nothing)
img=cv2.imread('abc.jpg')
while 1:
    th1=cv2.getTrackbarPos('yuzhi1','image')
    th2=cv2.getTrackbarPos('yuzhi2','image')
    edges=cv2.Canny(img,th1,th2)
    cv2.imshow('edges',edges)
    if cv2.waitKey(0)==ord('q'):
        break
cv2.destroyAllWindows()
