import cv2
import os
import face_recognition
import pickle 
import time

known_DIR = "known_faces"
#unknown_DIR = "unknown_faces"

TOLERNACE = 0.6
FRAME_THICKNESS = 3 
FONT_THICKNESS = 2
MODEL = "cnn"

video = cv2.VideoCapture(0)

print("loading faces")

known_faces = []
known_names = []

for name in os.listdir(known_DIR):
    for filename in os.listdir(f"{known_DIR}/{name}"):
        #image = face_recognition.load_image_file(f"{known_DIR}/{name}/{filename}")
        encoding = pickle.load(open(f"{name}/{filename}", "rb"))
        known_faces.append(encoding)
        known_names.append(name)

if len(known_names) > 0:
        next_id = max(known_names) + 1
else:
        next_id = 0

print("process unknown faces")

while True:
    #image = face_recognition.load_image_file(f"{unknown_DIR}/{filename}")
    rect,image = video.read()
    locations = face_recognition.face_locations(image , model=MODEL)
    encoding = face_recognition.face_encodings(image , locations)
    #image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
    
    for face_enoconding , face_location in zip(encoding,locations):
        results = face_recognition.compare_faces(known_faces, face_enoconding , TOLERNACE)
        match = None
        if True in results:
            match = known_faces[results.index(True)]
            print(f"Match Found : {match}")
        else:
                match = str(next_id)
                next_id += 1
                known_faces.append(match)
                known_names.append(face_enoconding)
                os.mkdir(f"{known_DIR}/{match}")
                pickle.dump(face_enoconding , open(f"{known_DIR}/{match}/{match}-{int(time.time())}.pkl" , "wb"))

        top_left = (face_location[3] , face_location[0])
        bottom_right = (face_location[1] , face_location[2])

        color = [0,255,0]
        cv2.rectangle(image,top_left , bottom_right , color, FRAME_THICKNESS)
        top_left = (face_location[3] , face_location[2])
        bottom_right = (face_location[1] , face_location[2] + 22)
        cv2.rectangle(image,top_left , bottom_right , color, cv2.FILLED)
            #cv2.putText(image,match,(face_location[3] +10, face_location[2] + 15) , cv2.FONT_HERSHEY_SIMPLEX , 0.5,(200,200,200),1)
    cv2.imshow(" " , image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break
