import face_recognition
import cv2
import numpy as np

camera = cv2.VideoCapture(0)  
'''picam2=Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()'''

if not camera.isOpened():
    print("Error: Failed to open the camera.")
    exit()

# Load a sample picture and learn how to recognize it
driver_image = face_recognition.load_image_file("Abhinav.jpg")
driver_face_encoding = face_recognition.face_encodings(driver_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    driver_face_encoding
]
known_face_names = [
    "Abhi"
]

while True:
    # Grab a single frame from the camera
    ret, frame = camera.read()
    '''frame = picam2.capture_array()
    frame = cv2.resize(frame, (450, 300))'''

    # Check if frame is captured successfully
    if frame is None:
        print("Error: Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and their encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame
    for face_encoding in face_encodings:
        # See if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the known name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        top, right, bottom, left = face_recognition.face_locations(rgb_frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
