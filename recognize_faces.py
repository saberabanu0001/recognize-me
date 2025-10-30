import face_recognition
import cv2
import os

# Directories
known_faces_directory = "/Users/saberabanu/All Drives/Personal/face-recognition/known-faces"
unknown_faces_directory = "/Users/saberabanu/All Drives/Personal/face-recognition/unknown-faces"

known_faces = []
known_names = []

# Load known faces
print("Loading known faces...")
for filename in os.listdir(known_faces_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(known_faces_directory, filename)
        image = face_recognition.load_image_file(path)  # ✅ correct function
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(filename)[0])

# Process unknown faces
print("Processing unknown faces...")
for filename in os.listdir(unknown_faces_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Checking {filename}...")
        path = os.path.join(unknown_faces_directory, filename)
        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in results:
                index = results.index(True)
                name = known_names[index]

            top, right, bottom, left = face_location
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)
