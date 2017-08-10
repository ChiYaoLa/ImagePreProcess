 # coding=utf-8
from PIL import Image
from pylab import *
from scipy.ndimage import filters

#im = array(Image.open('board.jpeg'))
im = array(Image.open('..\img\lena.jpg').convert('L'))

figure()
gray()
axis('off')
subplot(1, 4, 1)
axis('off')
title('original')
imshow(im)

for bi, blur in enumerate([2, 5, 10]):
  im2 = zeros(im.shape)
  im2 = filters.gaussian_filter(im, blur)
  im2 = np.uint8(im2)
  imNum=str(blur)
  subplot(1, 4, 2 + bi)
  axis('off')
  title('SD'+imNum)
  imshow(im2)

# #如果是彩色图像，则分别对三个通道进行模糊
# for bi, blur in enumerate([2, 5, 10]):
#  im2 = zeros(im.shape)
#  for i in range(3):
#    im2[:, :, i] = filters.gaussian_filter(im[:, :, i], blur)
#  im2 = np.uint8(im2)
#  subplot(1, 4,  2 + bi)
#  axis('off')
#  imshow(im2)

show()
