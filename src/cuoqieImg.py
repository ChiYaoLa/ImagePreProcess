# coding=utf-8
import cv2
import math
import os
# 图像错切
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

# 图像错切 操作 1是彩图 0是二值化  10是错切度数
InputImg = cv2.cv.LoadImage('..\img\lena.jpg', 0)
# InputImg = cv2.imread('..\img\lena.jpg', 1)
outImg = Warp(InputImg,10)
cv2.cv.ShowImage('input',InputImg)
cv2.cv.ShowImage('output',outImg)
cv2.cv.WaitKey(0)
cv2.cv.SaveImage('..\\aa.jpg',outImg)


