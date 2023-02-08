import cv2
import numpy as np
 

class detectFace():
    def __init__(self):
        self.CLASSIFIER = cv2.CascadeClassifier('Models//haarcascade_frontalface_default.xml')
        self.SIZE = 2

# We load the xml file
    def detectFace(self, image):
        image = cv2.resize(image, (image.shape[1] // self.SIZE, image.shape[0] // self.SIZE))
        results = self.CLASSIFIER.detectMultiScale(image) * self.SIZE
        for result in results:
            x1,y1,x2,y2 = result[0],result[1], result[0] + result[2],result[3] + result[1] 
            yield np.array([x1,y1,x2,y2 ])




# img = cv2.imread('photo_2022-09-24_16-00-50.jpg')
# detecter = detectFace()
# results = detecter.detectFace(img)

# for result in results:
#     print(result.shape)
#     startX, startY, endX, endY = result[0],result[1],result[2],result[3]
#     cv2.rectangle(img, (startX, startY), (endX, endY), (0,255,0), 2)

# cv2.imshow('imag',img)
# cv2.waitKey(0)


'''
classifier = 
size = 4
src=( r"rtsp://admin:sp@ce123@192.168.1.49:554")
webcam = cv2.VideoCapture(src) #Use camera 0

while True:
    (rval, im) = webcam.read()
    
    im = imutils.resize(im,height=500)
    # im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        # resized=cv2.resize(face_img,(150,150))
        # normalized=resized/255.0
        # reshaped=np.reshape(normalized,(1,150,150,3))
        # reshaped = np.vstack([reshaped])
        # result=model.predict(reshaped)
        # #print(result)
        
        # label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()
'''