# coding=utf-8
from PIL import Image
from pylab import *
# size
def ResizeImg(size):
    pil_im = Image.open('..\img\lena.jpg')
    pil_im = pil_im.resize(size)

    return pil_im


#resize 操作
subplot(121)
title('64*64')
axis('off')
imshow(ResizeImg((64,64)))
subplot(122)
title('128*128')
axis('off')
imshow(ResizeImg((128,128)))
# subplot(221)
# title('256*256')
# axis('off')
# imshow(ResizeImg((512,512)))
# subplot(222)
# title('512*512')
# axis('off')
# imshow(ResizeImg((128,128)))
show()
