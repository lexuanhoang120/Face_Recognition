"""
author : Xuan Hoang
Data : 23/9/2022
Function : find matching face in faces


"""


from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import logging
import numpy as np

class detectFace():
    
    def __init__(self):
        self.detector = MTCNN()
        self.required_size = (224, 224)

    def detectFace(self, image):
  
        # detect faces in the image
        results = self.detector.detect_faces(image)
        if results == []:
            return np.array([0])
        else:
            results =  np.array(results)
            # extract the bounding box from the first face
            for result in results:

                x1, y1, width, height = result['box']
                x2, y2 = x1 + width, y1 + height

                # resize pixels to the model size

                yield np.array([x1,y1,x2,y2])


# import imutils
# import cv2
# img = cv2.imread('istockphoto-1264660231-612x612.jpg')
# img = imutils.resize(img, width=1000)
# detector = detectFace()
# results = detector.detectFace(img)
# for result in results:
#     print(result.shape)
#     startX, startY, endX, endY = result[0],result[1],result[2],result[3]
#     cv2.rectangle(img, (startX, startY), (endX, endY), (0,255,0), 2)


# cv2.imshow('frame', img)
# cv2.waitKey(0)


