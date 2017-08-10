# coding=utf-8
# from __future__ import division
import cv2
import numpy as np
#这是遮罩效果  可以做mask 好屌啊 但是可惜有bug！
img1=cv2.imread('abc.jpg')
img2=cv2.imread('bbc.png')

rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]
img2gray=cv2.cvtColor(img2,cv2.COLOR_BAYER_BG2GRAY)
ret,mask=cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)
mask_inv=cv2.bitwise_not(mask)

img1_bg=cv2.bitwise_and(roi,roi,mask=mask)
img2_fg=cv2.bitwise_and(img2,img2,mask=mask_inv)
dst=cv2.add(img1_bg,img2_fg)
img1[0:rows,0:cols]=dst

cv2.imshow('img',img1)
