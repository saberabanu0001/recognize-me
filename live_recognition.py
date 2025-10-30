import face_recognition
import cv2
import os

# Load known faces
KNOWN_FACES_DIR = "known-faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # or "cnn" if you have a GPU

print("Loading known faces...")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    if name.endswith(('.jpg', '.jpeg', '.png')):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(name)[0])

print("Starting webcam...")

video = cv2.VideoCapture(0)

while True:
    ret, image = video.read()
    if not ret:
        break
# Resize frame for speed
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    encodings = face_recognition.face_encodings(rgb_small_frame, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        name = "Unknown"

        if True in results:
            match_index = results.index(True)
            name = known_names[match_index]

        # Scale back face locations since the frame was scaled
        top, right, bottom, left = [v * 4 for v in face_location]
        # Draw rectangle and label
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
        cv2.putText(image, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), FONT_THICKNESS)

    cv2.imshow("Live Face Recognition", image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()