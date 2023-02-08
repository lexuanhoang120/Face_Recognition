"""
author : Xuan Hoang
Data : 23/9/2022
Function : find matching face in faces


"""

# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os
import logging
import numpy as np



# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class identifyFace():
    def __init__(self ):
        # create a vggface model
        self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.required_size = (224,224)

    def getEmbeddings(self, faceDetected):
        faceDetected = Image.fromarray(faceDetected)
        faceDetected = faceDetected.resize(self.required_size)
        faceDetected = asarray(faceDetected)
        faceDetected = np.expand_dims(faceDetected, axis=0)
        # extract faces
        # convert into an array of samples
        samples = asarray(faceDetected, 'float32')
        # prepare the face for the model, e.g. center pixels
        sample = preprocess_input(samples, version=2)
        # perform prediction
        yhat = self.model.predict(sample)
        return yhat

	# determine if a candidate face is a match for a known face
    def isMatch(self, known_embedding, candidate_embedding, thresh=0.5):
        # calculate distance between embeddings
        score = cosine(known_embedding, candidate_embedding)
        
        # return [1,score] if score <= thresh else [0,score]
        if score <= thresh:
            print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
            # return [1,score]
        else:
            print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
            # return [0,score]



# import cv2
# # from detectFace import *
# import numpy as np
# # detector = detectFace()
# identifier = identifyFace()


# folder = "datatest/"
# file = "datatest/192.168.1.47_01_20221017115401440_FACE_SNAP.jpg"
# img0 = cv2.imread(file)
# embedding0 = identifier.getEmbeddings(img0)


# embeddings = []
# for fl in os.listdir(folder):
#     img = cv2.imread(folder + fl)
#     embedding = identifier.getEmbeddings(img)
#     identifier.isMatch(embedding0, embedding)
#     cv2.imshow('img' ,img)
#     cv2.waitKey(0)



# img = cv2.imread('datatest/192.168.1.47_01_20221017134253200_FACE_SNAP.jpg')
# img2 = cv2.imread('datatest/192.168.1.47_01_20221017134303677_FACE_SNAP.jpg')
# img3 = cv2.imread('datatest/192.168.1.47_01_20221017134320097_FACE_SNAP.jpg')
# img4 = cv2.imread('datatest\\192.168.1.47_01_20221017140211404_FACE_SNAP.jpg')
# img5 = cv2.imread('datatest\\192.168.1.47_01_20221017140332217_FACE_SNAP.jpg')
# img6 = cv2.imread('datatest\\192.168.1.47_01_20221017140432541_FACE_SNAP.jpg')
# img7 = cv2.imread('datatest/192.168.1.47_01_20221017134320097_FACE_SNAP.jpg')


# imageDeteced2 = detector.detectFace(img2)
# print(imageDeteced.shape, imageDeteced2.shape)
# logging.info(imageDeteced.shape)
# logging.info(imageDeteced2.shape)
# imageDeteced = np.expand_dims(imageDeteced, axis=0)
# imageDeteced2 = np.expand_dims(imageDeteced2, axis=0)
# print(imageDeteced.shape)
# embedding1 = identifier.getEmbeddings(img)
# embedding2 = identifier.getEmbeddings(img2) 
# embedding3 = identifier.getEmbeddings(img3)
# embedding4 = identifier.getEmbeddings(img4)
# embedding6 = identifier.getEmbeddings(img5)
# embedding3 = identifier.getEmbeddings(img6)
# identifier.isMatch(embedding1,embedding2)
# identifier.isMatch(embedding1,embedding3)
# b2 = np.load('embedingStaff.npy')

# c = np.load('codeStaff.npy')
# re = np.array([])
# for i in b2:
#     re = np.append(re, [cosine(i,embedding)],axis = 0)
# result = np.argmin(re)


# print(c[result])



# result = identifier.isMatch(embedding,embedding2)
# print('result : ', result)



# cv2.imshow('img',imageDeteced)
# cv2.waitKey(0)