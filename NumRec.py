# coding=utf-8
import numpy as np
import cv2

template = "tem2.jpg"

def sort_contours(cnts, method="left-to-right"):
  reverse = False
  i = 0
  if method == "right-to-left" or method == "bottom-to-top":
    reverse = True

  if method == "top-to-bottom" or method == "bottom-to-top":
    i = 1
  boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
  (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                    key=lambda b: b[1][i], reverse=reverse))
  return cnts, boundingBoxes

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
  dim = None
  (h, w) = image.shape[:2]
  if width is None and height is None:
    return image
  if width is None:
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    r = width / float(w)
    dim = (width, int(h * r))
  resized = cv2.resize(image, dim, interpolation=inter)
  return resized

def cv_show(name, img):
  cv2.imshow(name, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def Numfuction(predict_card):
	img = cv2.imread(template)
	image = cv2.imread(predict_card)

	ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ref = cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
	img, refCnts, hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	refCnts = sort_contours(refCnts,method="left-to-right")[0] 
	digits = {}

	for (i, c) in enumerate(refCnts):
	  (x, y, w, h) = cv2.boundingRect(c)
	  roi = ref[y:y + h, x:x + w]
	  roi = cv2.resize(roi, (57, 88))
	  digits[i] = roi
	################################################################
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	orimg = resize(image, width=900)
	image = resize(image, width=300)

	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT, rectKernel)
	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	img, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	cnts = []
	maxcnts = 0
	idxcnts = 0
	# clean
	for n in range(len(threshCnts)):
		if(len(threshCnts[n])>15):
			if(cv2.minAreaRect(threshCnts[n])[1][1] >6):
				cnts.append(threshCnts[n])

	for n in range(len(cnts)):
		if(cv2.contourArea(cnts[n])> maxcnts ):
			maxcnts = cv2.contourArea(cnts[n])
			idxcnts = n
	del cnts[idxcnts]

	# cv2.drawContours(image, cnts, -1, (0, 0, 255), 1)
	# cv_show("img", image)

	longcnts=0
	idxcnts=0
	for n in range(len(cnts)):
		if(cv2.minAreaRect(cnts[n])[1][0]>longcnts):
			longcnts = cv2.minAreaRect(cnts[n])[1][0]
			idxcnts = n

	# cv2.drawContours(image, cnts, idxcnts, (0, 0, 255), 1)
	# cv_show("img", image)

	rect = cv2.minAreaRect(cnts[idxcnts])
	# print(rect[2])
	###################################################################################
	box = cv2.boxPoints(rect)
	box = np.int0(box)

	target = orimg[box[1][1]*3-6:box[0][1]*3+6, box[1][0]*3-6:box[2][0]*3+15]
	ortarget = target.copy()
	orgray=cv2.cvtColor(ortarget,cv2.COLOR_BGR2GRAY)
	_, orgray=cv2.threshold(orgray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

	gray=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
	_, gray=cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

	img, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	# cv2.drawContours(ortarget,contours,-1,(0,0,255),1)
	# cv_show("img", ortarget)

	for n in range(1, len(contours)):
		for m in range(1, len(contours)):
			if cv2.minAreaRect(contours[m])[0][0]>cv2.minAreaRect(contours[m-1])[0][0]:
				temp = contours[m-1]
				contours[m-1] = contours[m]
				contours[m] = temp
	newcontours = []
	for n in range(1, len(contours)):
		if((cv2.minAreaRect(contours[n-1])[0][0]-cv2.minAreaRect(contours[n])[0][0])>5):
			newcontours.append(contours[n-1])
		else:
			break

	groupOutput = []
	output = []
	for c in newcontours:
	  (x, y, w, h) = cv2.boundingRect(c)
	  roi = orgray[y:y + h, x:x + w]
	  roi = cv2.resize(roi, (57, 88))

	  scores = []
	  for (digit, digitROI) in digits.items():
	    result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
	    (_, score, _, _) = cv2.minMaxLoc(result)
	    scores.append(score)

	  groupOutput.append(str(np.argmax(scores)))
	output.extend(groupOutput)
	output.reverse()

	return output

if __name__=='__main__':
	output = Numfuction('test2.jpg')
	print ("Num #: {}".format("".join(output)))
