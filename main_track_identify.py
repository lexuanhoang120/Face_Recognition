import configparser
import datetime
import time

import cv2
from imutils.video.videostream import VideoStream

from packages.detectFaceCNN4 import detectFace
from packages.tracking_objects import CentroidTracker, draw_bounding_boxes
import logging
import imutils
# PADDING = 10
# THRESHOLD_DISTANCE = 100
# SIZE_SCALE = 2
# MAX_DISAPPEARED = 15
#
# # SOURCE = r"rtsp://admin:space123@192.168.1.49:554"
# SOURCE = 0
# SHOW = True
logging.basicConfig(filename='app.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    level=logging.INFO)

def tracking_camera(source, is_showed=True):
    connection = VideoStream(source).start()
    detecter = detectFace()
    tracker = CentroidTracker(threshold_distance=100, max_disappeared=10)
    while True:
        frame = connection.read()
        # check signal from connection
        if frame is None:
            connection.stop()
            time.sleep(0.1)
            logging.error('Disconnected Camera')
            connection.start()
            continue
        # do face tracking and ignore facial landmarks
        bounding_boxes,scores_mask = detecter.detectFace(frame)
        # update face tracker
        objects, code,rect_identify = tracker.update(bounding_boxes,scores_mask,frame)
        # for id_face in id_bounding_boxes_face_tracked:
        #     startX, startY, endX, endY = bounding_boxes[id_face]
        #     saving_time = datetime.datetime.now().strftime('%m%d%Y%H%M%S%f')
        #     cv2.imwrite(f"images/face/{saving_time}.jpg",
        #                 frame[max(startY - PADDING, 0):min(endY + PADDING, frame.shape[0]),
        #                 max(startX - PADDING, 0):min(endX + PADDING, frame.shape[1])])
        #     cv2.imwrite(f"images/context/{saving_time}.jpg",
        #                 frame)
        if is_showed:
            frame = draw_bounding_boxes(frame, rect_identify)
            
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "{}".format(code[objectID])
                cv2.putText(frame, text, (centroid[0], centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                # show the output frame
            frame = imutils.resize(frame,width = 1000)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
        else:
            continue
    connection.stop()


if __name__ == "__main__":
    src = r"rtsp://admin:sp@ce123@192.168.1.61:554"
    # src = 0
    tracking_camera(source=src, is_showed=True)
