# coding=utf-8
import cv2
import numpy as np

# #查找模块所有函数和属性 用dir
# funcs=dir(cv2)
# index=0
# for i in funcs:
#     print i
#     index += 1
# print index



#缩放图片  OK'的
img = cv2.imread('abc.jpg')
#直接根据倍数缩放
res=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
print img.shape
print img.shape[:2]
# #直接根据尺寸缩放
# height,width = img.shape[:2]
# res=cv2.resize(img,(2*height,2*width),interpolation=cv2.INTER_CUBIC)

while(1):
    cv2.imshow('res',res)
    cv2.imshow('img',img)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()



