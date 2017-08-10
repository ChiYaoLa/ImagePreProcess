# coding=utf-8
import cv2
#摄像头 抓取一帧
cap=cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    # #彩色
    # cv2.imshow('frame',frame)

    #灰色
    gray = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2GRAY)  #是属性有问题吗 这个真是奇怪！！明早看
    cv2.imshow('frame',gray)

    if cv2.waitKey(0)==ord('q'):
        break
cap.release()
