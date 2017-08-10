# coding=utf-8
import os
print os.path.join('C:\local','lena.jpg')
path = os.path.join('C:\local','lena.jpg')
print os.path.dirname(path)
print os.path.basename(path) #风动旛动
print os.path.splitext(path)
print os.path.split(path)

for i in  range(0,20):
    print i
