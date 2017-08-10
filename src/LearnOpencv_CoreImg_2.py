# coding=utf-8
# from __future__ import division
import cv2
import numpy as np

img = cv2.imread('abc.jpg')

# print img.item(0,0,0)  #像素值
# img.itemset((0,0,0),0)
# print img.shape
# print img.size
# print img.dtype

# #局部替换
# ball = img[280:340,330:390]
# img[273:333,100:160]=ball
# cv2.imshow('img',img)
# cv2.waitKey(0)

#操作通道
# (b,g,r)=cv2.split(img)  #b g r是三个二维矩阵  mv形参是b g r三个矩阵
# print (b,g,r)
# img=cv2.merge((b,g,r))  #奇怪了，这里不能写 b,g,r 必须写元组
# cv2.imshow('img',img)
# cv2.waitKey(0)


# #直接在通道上 变颜色
# img[:,:,0]=0
# cv2.imshow('img',img)
# cv2.waitKey(0)

#图像加法
img2=cv2.imread('bbc.png')
imgshape=img.shape
# fx=float(img2.shape[0])/float(img.shape[0])  #精确除法的两种方法之一
# fy=float(img2.shape[1])/float(img.shape[1])
# cv2.resize((img2,0,img2,fx,fy))
# cv2.resize(img2,(imgshape[0],imgshape[1]),img2, cv2.INTER_CUBIC)   #resize 操作反正没成功的  手动的
img3=cv2.addWeighted(img,0.5,img2,0.5,1)
cv2.imshow('img',img3)
cv2.imwrite('haomeidetu.jpg',img3)
cv2.waitKey(0)
