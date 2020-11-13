import cv2
import numpy as np
def Numfuction(image,height,wide):
    img_1=cv2.imread('tem2.jpg',0)
    num=[]
    for i in range(10):
        img_2=img_1[0:126,80*i:80*(i+1)]

        img_2=cv2.resize(img_2,(int(wide),int(height)))

        w, h = img_2.shape[::-1]
        res = cv2.matchTemplate(image, img_2, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res >= threshold)
        if loc !=None:
            num.append(i)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return image,num
