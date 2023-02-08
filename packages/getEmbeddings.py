"""
author : Xuan Hoang
Data : 23/9/2022
Function : find matching face in faces


"""


# from detectFaceCNN import *
from packages.identifyFace import identifyFace
import os
import numpy as np
import cv2
def get_embedding():
    ABSOLUTE_DIR = 'C:/Users/VTCODE/Documents/identify_human/sources/dataset/'
    identifier = identifyFace()
    folders = os.listdir(ABSOLUTE_DIR)
    embeddingStaff = np.array([])
    codeStaff = np.array([])
    for folder in folders:
        files = os.listdir(ABSOLUTE_DIR + "/" + folder)
        
        for file in files:
            try:
                pathFile = ABSOLUTE_DIR + '/'+folder+'/'+file
                img = cv2.imread(pathFile)
                
                if embeddingStaff.size == 0:
                    embeddingStaff = identifier.getEmbeddings(img)
                    
                else:
                    embeddingStaff = np.append(embeddingStaff,identifier.getEmbeddings(img), axis= 0 )
                codeStaff = np.append(codeStaff,[folder], axis = 0)
            except:
                pass
    np.save('sources//embeddingNPY//embedingStaff.npy',embeddingStaff)
    np.save('sources//embeddingNPY//codeStaff.npy',codeStaff)
    return embeddingStaff,codeStaff























