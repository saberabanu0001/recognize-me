import face_recognition
import cv2
import os


known_faces_directory = "/Users/saberabanu/All Drives/Personal/face-recognition/known-faces"
unknown_faces_directory = "/Users/saberabanu/All Drives/Personal/face-recognition/unknown-faces"

known_faces = []
known_names = []

for filename in os.listdir(known_faces_directory):
    images = face_recognition.load_image_files(f"known_faces_directory/{known_faces}")
    encoding = face_recognition.face_encodings(images)[0]
    known_faces.append(encoding)
    known_faces.append(os.path.splitext(filename)[0])

    print("Processing known faces.........")

for filename in os.listdir(known_faces_directory):
    print("Checking filename....")
    images = face_recognition.load_image_files(f"known_faces_directory/{known_faces}")
    locations = face_recognition.face_locations(images)[0]
    encodings = face_recognition.face_encodings(images, locations)



    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for face_encoding, face_location in zip(encodings, locations):
    results = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)

    name = "unknown"

    if True in results:
        match_index = results.index[True]
        name = known_names[match_index]

    #draw boxes
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(image, name(left,top-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255, 25, 255), 2)