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
import time
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import logging
import cv2

class findFace():
    def __init__(self ):
        # create a vggface model
        self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.required_size = (224,224)
        

    def getEmbedding(self, faceDetected):
        # logging.info('Call getEmbedding function')

        faceDetected = cv2.resize(faceDetected,self.required_size)
        faceDetected = np.expand_dims(faceDetected, axis=0)
        # extract faces
        # convert into an array of samples
        samples = asarray(faceDetected, 'float32')
        # prepare the face for the model, e.g. center pixels
        sample = preprocess_input(samples, version = 2)
        # perform prediction
        embedding = self.model.predict(sample)
        # logging.info('Get embedding')
        return embedding

	# determine if a candidate face is a match for a known face
    def findFace(self, embedding, embeddingFaces):
        tempDistance = np.array([])
        for embeddingFace in embeddingFaces:
            tempDistance = np.append(tempDistance, [cosine(embeddingFace,embedding)], axis=0)
        
        return [np.argmin(tempDistance), tempDistance[np.argmin(tempDistance)]]


        # # calculate distance between embeddings
        # score = cosine(known_embedding, candidate_embedding)
        
        # return [1,score] if score <= thresh else [0,score]
        # if score <= thresh:
        #     # print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        #     return [1,score]
        # else:
        #     # print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
        #     return [0,score]





# a = findFace()
# import cv2
# import detectFaceCNN2

# img = cv2.imread('photo_2022-08-04_16-14-16.jpg')
# # cv2.imshow('hf',img)
# # cv2.waitKey(0)
# # detecter = human_mask()
# # i = 0
# # bb = np.zeros((1, 4), dtype=np.int32)
# # (locs, preds) = detecter.detect_and_predict_mask(img)
# # for (box,_) in zip(locs, preds):
# # # detectedImg = img[]
# #     (startX, startY, endX, endY) = box
# #     bb[i][0] = startX
# #     bb[i][1] = startY
# #     bb[i][2] = endX
# #     bb[i][3] = endY         
# #     detectedImg = img[bb[i][1]:bb[i]
# #                 [3], bb[i][0]:bb[i][2], :]


# detecter = detectFaceCNN2.detectFaceCNN()
# detectedImg = detecter.detectFaceCNN(img)

# # cv2.imshow('hf',detectedImg)
# # cv2.waitKey(0)

# embedding1 = a.getEmbedding(detectedImg)
# print(embedding1.shape)


# # b1 = np.array([[1,2,3]])

# b2 = np.load('embedingStaff.npy')

# c = np.load('codeStaff.npy')

# re = a.findFace(embedding1,b2)

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