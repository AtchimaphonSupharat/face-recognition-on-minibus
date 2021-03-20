import requests
import urllib.parse
import cv2
import face_recognition
import numpy as np
import imutils
import time
import sys
import os
from imutils.object_detection import non_max_suppression
from time import sleep
current_path = os.getcwd()
CONFIDENCE = 0.5 
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
counter = 0

url_line = 'https://notify-api.line.me/api/notify'
line_token = 'IO76yqQoVg41HxAf4gCCwbB5B4nV9Td3GnoHKB8GWLm'
# Load the Haar Cascade 
cascPath = 'D:/project/projectyolo/haarcascade_frontalface_default.xml'
# Create the Haar Cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# the neural network configuration
config_path = "D:/project/projectyolo/config_object/yolov3.cfg"
# the YOLO net weights file
weights_path = "D:/project/projectyolo/config_object/yolov3.weights"
# weights_path = "weights/yolov3-tiny.weights"

# loading all the class labels (objects)
labels = open("D:/project/projectyolo/config_object/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

##########
#image for detec
path_name = "D:/project401/recog/test/33.jpg"
image = cv2.imread(path_name)
img = cv2.imdecode(np.fromfile(path_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

#print (image)
#database image
lmm_image = face_recognition.load_image_file("D:/project401/recog/database/4.jpg")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)


known_faces = [
    lmm_face_encoding
]
scale_percent = 50 # percent of original size #เดิม50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
image1 = image
h, w = image.shape[:2]
# create 4D blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
#print("image.shape:", image.shape)
#print("blob.shape:", blob.shape)


# sets the blob as the input of the network
net.setInput(blob)
# get all the layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# feed forward (inference) and get the network output
# measure how much it took in seconds
layer_outputs = net.forward(ln)

font_scale = 1
thickness = 1
boxes, confidences, class_ids = [], [], []

# loop over each of the layer outputs
for output in layer_outputs:
    # loop over each of the object detections
    for detection in output:
        # extract the class id (label) and confidence (as a probability) of
        # the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)       
        confidence = scores[class_id]
        if confidence > CONFIDENCE:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            overlay = image.copy()
# loop over the indexes we are keeping
for i in range(len(boxes)): 
    color = [int(c) for c in colors[class_ids[i]]]
    xxx = image1[y:y + h,x:x + w]
    if len(xxx) != 0:
        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        overlay = image.copy()
        gray = cv2.cvtColor(xxx, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_names = []
        for face_encoding in face_encodings:  
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            name = None
            if match:
                name = "Driver"            
            face_names.append(name)
            
            # Label the results
        for (x, y, w, h), name in zip(face_locations, face_names):
            if not name:
                continue
            cv2.rectangle(xxx, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop_img = image[x+6:w+6, h+6:y+6]
            if(name == "Face"):
                #cv2.putText(image, name, (y, x), cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                cv2.imwrite(current_path + filename+ "people"+str(counter)+".png",crop_img)
                counter = counter + 1 
            if(name == "Driver"):
                cv2.putText(image, name, (y, x), cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
               
for name in face_names:
    if(name == "Driver"):
                    cv2.putText(image, name, (y, x), cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                    #cv2.imwrite(current_path + filename+ "people"+str(counter)+".png",crop_img)
                    file_img = {'imageFile': open('D:/project401/recog/database/4.jpg', 'rb')}
                    #file_img1 = {'imageFile': open('D:/project401/recog/result'+filename+ "result"+".png", 'rb')}
                    msg = ({'message': 'He is Driver'})
                    #msg1 = ({'message': file_name+' processing image'})
                    LINE_HEADERS = {"Authorization":"Bearer "+line_token}
                    session = requests.Session()
                    session_post = session.post(url_line, headers=LINE_HEADERS, files=file_img, data=msg)
                    #session_post1 = session.post(url_line, headers=LINE_HEADERS, files=file_img1,data=msg1)
                    print(session_post.text) 
                    #print(session_post1.text)
    elif(lmm_face_encoding != image):                  
                    #file_img = {'imageFile': open('D:/project401/recog/database/4.jpg', 'rb')}
                    file_img = {'imageFile': open('D:/project401/recog/not.jpg', 'rb')}
                    file_img1 = {'imageFile': open('D:/project401/recog/result'+filename+ "result"+".png", 'rb')}
                    msg = ({'message': 'He is not Driver'})
                    msg1 = ({'message': file_name+' processing image'})
                    LINE_HEADERS = {"Authorization":"Bearer "+line_token}
                    session = requests.Session()
                    session_post = session.post(url_line, headers=LINE_HEADERS, files=file_img, data=msg)
                    session_post1 = session.post(url_line, headers=LINE_HEADERS, files=file_img1,data=msg1)
                    print(session_post.text) 
                    print(session_post1.text)      

if(lmm_face_encoding != image):                  
                    #file_img = {'imageFile': open('D:/project401/recog/database/4.jpg', 'rb')}
                    file_img = {'imageFile': open('D:/project401/recog/not.jpg', 'rb')}
                    file_img1 = {'imageFile': open('D:/project401/recog/result'+filename+ "result"+".png", 'rb')}
                    msg = ({'message': 'He is not Driver'})
                    msg1 = ({'message': file_name+' processing image'})
                    LINE_HEADERS = {"Authorization":"Bearer "+line_token}
                    session = requests.Session()
                    session_post = session.post(url_line, headers=LINE_HEADERS, files=file_img, data=msg)
                    session_post1 = session.post(url_line, headers=LINE_HEADERS, files=file_img1,data=msg1)
                    print(session_post.text) 
                    print(session_post1.text)     

rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
faces = non_max_suppression(rects, probs=None, overlapThresh=0.6)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('xxx', image)
    cv2.waitKey(0)
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org1 = (50, 100)  
# fontScale 
fontScale = 0.8
# pink color in BGR 
color = (255, 0, 255) 
# Line thickness of 2 px 
thickness = 2
cv2.imwrite('D:/project401/recog/result'+filename+ "result"+".png",image)
#################################################
# Output img with window name as 'image' 
cv2.imshow('image', image)
#cv2.imshow('thresh1', thresh1)    
# Maintain output window utill 
# user presses a key 
cv2.waitKey(0)          
# Destroying present windows on screen 
cv2.destroyAllWindows()  




