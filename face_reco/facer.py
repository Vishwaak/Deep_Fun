import cv2
import os
import face_recognition

known_DIR = "known_faces"
unknown_DIR = "unknown_faces"

TOLERNACE = 0.6
FRAME_THICKNESS = 3 
FONT_THICKNESS = 2
MODEL = "cnn"

print("loading faces")

known_faces = []
known_names = []

for name in os.listdir(known_DIR):
    for filename in os.listdir(f"{known_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{known_DIR}/{name}/{filename}")
        encoding  = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("process unknown faces")

for filename in os.listdir(unknown_DIR):
    image = face_recognition.load_image_file(f"{unknown_DIR}/{filename}")
    locations = face_recognition.face_locations(image , model=MODEL)
    encoding = face_recognition.face_encodings(image , locations)
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
    
    for face_enoconding , face_location in zip(encoding,locations):
        results = face_recognition.compare_faces(known_faces, face_enoconding , TOLERNACE)
        match = None
        if True in results:
            match = known_faces[results.index(True)]
            print(f"Match Found : {match}")

            top_left = (face_location[3] , face_location[0])
            bottom_right = (face_location[1] , face_location[2])

            color = [0,255,0]
            cv2.rectangle(image,top_left , bottom_right , color, FRAME_THICKNESS)

            top_left = (face_location[3] , face_location[2])
            bottom_right = (face_location[1] , face_location[2] + 22)
            cv2.rectangle(image,top_left , bottom_right , color, cv2.FILLED)
            #cv2.putText(image,match,(face_location[3] +10, face_location[2] + 15) , cv2.FONT_HERSHEY_SIMPLEX , 0.5,(200,200,200),1)
    cv2.imshow(filename , image)
    cv2.waitKey(1000)