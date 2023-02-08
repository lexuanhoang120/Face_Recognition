from collections import OrderedDict
import threading
import cv2
import numpy as np
from scipy.spatial import distance as dist
from packages.postAlert import post_to_of1
from packages.findFace2 import findFace
from packages.insert_information2 import *
from packages.alertCheck import AlertCheck
from packages.getEmbeddings2 import get_embedding
from tensorflow.keras.models import load_model
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



def draw_bounding_boxes(frame, bounding_boxes):
    for bbd in bounding_boxes:
        x1, y1, x2, y2 = int(bbd[0]), int(bbd[1]), int(bbd[2]), int(bbd[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255 , 0), 4)
    return frame


class CentroidTracker:
    def __init__(self, threshold_distance=100, max_disappeared=15, frame_tracked=3, threshold=0.3, ):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.codes = OrderedDict()
        self.disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.max_disappeared = max_disappeared
        self.threshold_distance = threshold_distance
        self.frame_tracked = frame_tracked
        self.threshold = threshold
        self.code_alert = set()
        self.code_accuracy = set()
        self.identify = findFace()
        self.rect_identify = np.array([])
        self.embedding_staff,self.code_staff = get_embedding()
        # self.code_staff = np.load('sources\embeddingNPY\codeStaff.npy')
        # self.embedding_staff = np.load('sources\embeddingNPY\embedingStaff.npy')
        threading.Thread(target=self.alert, args=()).start()
        threading.Thread(target=self.post_1office_insert_information,args=()).start()

    def alert(self):
        alert = AlertCheck()
        while True:
            time.sleep(0.25)
            for code in self.code_alert.copy():
                alert.alert(code)
                self.code_alert.remove(code)

    def post_1office_insert_information(self):
        # database = connect_database()
        while True:
            time.sleep(0.25)
            for code in self.code_accuracy.copy():
                insert_information(code[0],code[1])
                post_to_of1(code[0])
                self.code_accuracy.remove(code)


    def register(self, rect,score_mask, frame):
        # when registering an object we use the next available object
        # ID to store the centroid
        startX, startY, endX, endY = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        image_detected = frame[startY:endY, startX:endX]
        embedding_image = self.identify.getEmbedding(image_detected)
        result = self.identify.findFace(embedding_image, self.embedding_staff)
        code = self.code_staff[result[0]]

        accuracy = self.threshold - 0.1 +0.1*(1-score_mask)
        if result[1] < accuracy:
            save_image(image_detected, code)
            self.rect_identify = np.append(self.rect_identify,[startX, startY, endX, endY])
            self.code_alert.add(code)
            self.code_accuracy.add((code,result[1]))
            self.objects[self.nextObjectID] = self.locating_centroid(rect)
            self.codes[self.nextObjectID] = code
            self.disappeared[self.nextObjectID] = 0
            self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.codes[objectID]

    @staticmethod
    def locating_centroids(bounding_boxes):
        if bounding_boxes.size == 0:
            return []
        return np.c_[
            (bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2,
            (bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2,
        ].astype("int")

    @staticmethod
    def locating_centroid(bounding_box):
        if bounding_box.size == 0:
            return []
        return np.c_[
            (bounding_box[0] + bounding_box[2]) / 2,
            (bounding_box[1] + bounding_box[3]) / 2,
        ][0].astype("int")

    def update(self, rects, scores_mask,frame):
        self.rect_identify = np.array([])
        # check to see if the list of input bounding box rectangles
        # is empty
        if rects.size == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects, self.codes,self.rect_identify
            # otherwise, are are currently tracking objects so we need to
            # try to match the input centroids to existing object
            # centroids
        # initialize an array of input centroids for the current frame
        inputCentroids = self.locating_centroids(rects)
        # inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # bounding_boxes = np.zeros((len(rects), 4), dtype="int")
        # # loop over the bounding box rectangles
        # for (i, (startX, startY, endX, endY)) in enumerate(rects):
        #     # use the bounding box coordinates to derive the centroid
        #     cX = int((startX + endX) / 2.0)
        #     cY = int((startY + endY) / 2.0)
        #     inputCentroids[i] = (cX, cY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            self.nextObjectID = 0
            for i in range(0, len(inputCentroids)):
                self.register(rect=rects[i],score_mask = scores_mask[i], frame=frame)
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            inThreshold = np.where((D.min(axis=1) < self.threshold_distance) == True)[0]
            if inThreshold.size == 0:
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            else:
                for (row, col) in zip(rows, cols):
                    # if we have already examined either the row or
                    # column value before, ignore it
                    # val
                    if row in usedRows or col in usedCols:
                        continue
                    if row in inThreshold:
                        # otherwise, grab the object ID for the current row,
                        # set its new centroid, and reset the disappeared
                        # counter
                        objectID = objectIDs[row]

                        self.objects[objectID] = inputCentroids[col]
                        self.disappeared[objectID] = 0

                        self.rect_identify = np.append(self.rect_identify,rects[col])

                        # indicate that we have examined each of the row and
                        # column indexes, respectively
                        usedRows.add(row)
                        usedCols.add(col)
                # compute both the row and column index we have NOT yet
                # examined
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(rect=rects[col],score_mask=scores_mask[col], frame=frame)

            # return the set of trackable objects
        return self.objects, self.codes,np.reshape(self.rect_identify,(-1,4))



