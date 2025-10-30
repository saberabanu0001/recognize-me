import face_recognition
import cv2
import os


known_faces_directory = "/Users/saberabanu/All Drives/Personal/face-recognition/known-faces"
unknown_faces_directory = "/Users/saberabanu/All Drives/Personal/face-recognition/unknown-faces"

known_faces = []
unknown_faces = []

for filename in os.listdir(known_faces_directory):
    images = face_recognition.load_image_files(f"known_faces_directory/{known_faces}")


