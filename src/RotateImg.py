# coding=utf-8
from PIL import  Image
from pylab import *

def Rotate(angle):
    pil_im = Image.open('..\img\lena.jpg')
    pil_im = pil_im.rotate(angle)
    return pil_im

#旋转操作
title("45 angle")
axis('off')
imshow(Rotate(45))
show()
