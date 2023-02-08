"""
author : Xuan Hoang
Data : 23/9/2022
Function : find matching face in faces


"""



# face verification with the VGGFace2 model
# from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
import cv2
import time
# from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import logging
import tensorflow as tf
# from keras_vggface.vggface import VGGFace

class findFace():
    def __init__(self ):
        # create a vggface model
        # self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.model  = tf.keras.models.load_model('sources/model_embedding')
        self.required_size = (224,224)
        

    def getEmbedding(self, faceDetected):
        # logging.info('Call getEmbedding function')
        # faceDetected = Image.fromarray(faceDetected)
        faceDetected = cv2.resize(faceDetected,self.required_size)

        face_array = np.expand_dims(faceDetected, axis=0,)


        # # cv2.imshow('fdf',faceDetected)
        # # cv2.waitKey(0)

        # faceDetected = faceDetected.resize(self.required_size)  

        # # extract faces

        # # convert into an array of samples
        face_array = asarray(face_array, 'float32')
        # # prepare the face for the model, e.g. center pixels
        face_array = preprocess_input(face_array, version = 2)
        # # perform prediction
        # print(faceDetected.shape)

        embedding = self.model.predict(face_array)
        # logging.info('Get embedding')
        return embedding

	# determine if a candidate face is a match for a known face
    def findFace(self, embedding, embeddingFaces):
        tempDistance = np.array([])
        embedding = np.resize(embedding,(2048,))

        for index in range(embeddingFaces.shape[0]):

            embeddingFace = np.resize(embeddingFaces[index],(2048,))
            if tempDistance.size == 0:
                tempDistance = np.array([cosine(embeddingFace,embedding)])
            else:
                tempDistance = np.append(tempDistance, [cosine(embeddingFace,embedding)] , axis=0)
    
        return [np.argmin(tempDistance), tempDistance[np.argmin(tempDistance)]]





# a = findFace()
# import cv2
# # from detectFaceCNN3 import detectFace
#
# img = cv2.imread('D:\Documents\CTidentifystaff\datatest/192.168.1.47_01_20221017114633897_FACE_SNAP.jpg')
# print(img.shape)
# re = a.getEmbedding(img)
# print(re)
# img = cv2.resize(img,(224,224))
# print(img.shape)

# embedding = a.getEmbedding(img)
# print(embedding.shape)


# # embedding1 = a.getEmbedding(detectedImg)
# # print(embedding1.shape)


# # b1 = np.array([[1,2,3]])

# b2 = np.load('sources/embeddingNPY/embedingStaff.npy')

# c = np.load('sources/embeddingNPY/codeStaff.npy')
# print(b2.shape)
# re = a.findFace(embedding,b2)

# print(c[re[0]])
# print(re)
# print(re[1])





# print(re)

# import cv2
# from detectFace import *
# import numpy as np
# detector = detectFace()
# identifier = identifyFace()
# img = cv2.imread('photo_2022-09-16_11-54-35.jpg')
# img2 = cv2.imread('photo_2022-08-04_16-14-16.jpg')
# imageDeteced = detector.detectFace(img)
# imageDeteced2 = detector.detectFace(img2)
# # print(imageDeteced.shape, imageDeteced2.shape)
# # logging.info(imageDeteced.shape)
# # logging.info(imageDeteced2.shape)
# # imageDeteced = np.expand_dims(imageDeteced, axis=0)
# # imageDeteced2 = np.expand_dims(imageDeteced2, axis=0)
# # print(imageDeteced.shape)
# embedding = identifier.getEmbeddings(imageDeteced)
# embedding2 = identifier.getEmbeddings(imageDeteced2)
# result = identifier.isMatch(embedding,embedding2)
# print('result : ', result)
# # cv2.imshow('img',imageDeteced)
# # cv2.waitKey(0)