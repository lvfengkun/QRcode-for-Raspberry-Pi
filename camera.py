# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import serial
from Numpre import Numfuction
from pyzbar import pyzbar
import RPi.GPIO as GPIO  
import time
'''
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (480, 255)
camera.framerate = 32
# camera.saturation = 50
camera.brightness = 60
camera.sharpness  = 100
camera.hflip = True
camera.vflip = True
rawCapture = PiRGBArray(camera, size=(480, 255))
# allow the camera to warmup
time.sleep(0.1)'''

#ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
#print(ser.isOpen())
drawing = False
mode = True
start = (-1, -1)
GPIO.setmode(GPIO.BCM)
#GPIO.setup(13,GPIO.OUT) 
def decodeDisplay(image):
    barcodes = pyzbar.decode(image)
    height=0
    wide=0
    for barcode in barcodes:
        # 提取二维码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        height=h
        wide=w
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 提取二维码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)

        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
       
        
        GPIO.setup(13,GPIO.OUT) 
        GPIO.output(13, GPIO.HIGH)  
        
    
          
    return image,height,wide


cv2.namedWindow('image')

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera)
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        #cv2.imshow("camera_1", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im,h,w = decodeDisplay(gray)
        
        height=h*0.18
        wide=w*0.12
        if h!=0: 
            img,output = Numfuction(im,height,wide)
            #print("Num: {}".format("".join(output)))
            GPIO.setmode(GPIO.BCM)  
            
            if(output!=[]):
                print(output)
                GPIO.setup(19, GPIO.OUT)
                GPIO.output(19,GPIO.HIGH)   
                   
            else:
                GPIO.setup(19, GPIO.OUT)
                GPIO.output(19,GPIO.LOW)   
                  
                
            im=img
        
            
            #time.sleep(0.1)
        if(h==0):
            img_2=cv2.imread("QRcode.jpg",0)
            #cv2.imwrite("img_2.jpg",im)
            flag=0
            w_1, h_1 = img_2.shape[::-1]
            res = cv2.matchTemplate(im, img_2, cv2.TM_CCOEFF_NORMED)
            threshold = 0.3
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(im, pt, (pt[0] + w_1, pt[1] + h_1), (0, 0, 255), 2)
                flag=1
            if(flag==1):
                GPIO.setup(13,GPIO.OUT) 
                GPIO.output(13, GPIO.HIGH)
            else:
                GPIO.setup(13,GPIO.OUT) 
                GPIO.output(13, GPIO.LOW)
             
        cv2.imshow("camera", im)
        #cv2.imwrite("image/image.jpg",im)
        cv2.waitKey(1)
    
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        if key == ord("q"):
                break
