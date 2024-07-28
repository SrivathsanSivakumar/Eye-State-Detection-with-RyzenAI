### Open webcam and ensure MTCNN identifies eyes correctly
### End program using CTRL+C or CMD+C

import cv2, dlib
import numpy as np
from facenet_pytorch import MTCNN


cam_capture = cv2.VideoCapture(0)
detector = MTCNN(select_largest=False, post_process=False)
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faces = dlib.full_object_detections()

while True:
    ret, frame = cam_capture.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = detector.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            box = box.astype(int)
            det = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            face_landmarks = [(item.x, item.y) for item in sp(rgb_frame, det).parts()]
            face_landmarks = np.array(face_landmarks)

            # Calculate dimensions and position for the left eye
            left_eye_length = 2 * (face_landmarks[39, 0] - face_landmarks[36, 0])
            left_eye_width = 4 * (face_landmarks[41, 1] - face_landmarks[37, 1])
            left_eye_startx = face_landmarks[36, 0] - 0.3 * left_eye_length
            left_eye_starty = face_landmarks[37, 1] - 0.3 * left_eye_width

            # Calculate dimensions and position for the right eye
            right_eye_length = 2 * (face_landmarks[45, 0] - face_landmarks[42, 0])
            right_eye_width = 4 * (face_landmarks[47, 1] - face_landmarks[43, 1])
            right_eye_startx = face_landmarks[42, 0] - 0.3 * right_eye_length
            right_eye_starty = face_landmarks[43, 1] - 0.3 * right_eye_width
                   
            left_eye_img = rgb_frame[int(left_eye_starty):int(left_eye_starty + left_eye_width), int(left_eye_startx):int(left_eye_startx + left_eye_length)]
            if left_eye_img.size != 0:
                cv2.rectangle(frame, (int(left_eye_startx), int(left_eye_starty)), (int(left_eye_startx + left_eye_length), int(left_eye_starty + left_eye_width)), (0, 255, 0), 2)

            # Process and save right eye region
            right_eye_img = rgb_frame[int(right_eye_starty):int(right_eye_starty + right_eye_width), int(right_eye_startx):int(right_eye_startx + right_eye_length)]
            if right_eye_img.size != 0:
                cv2.rectangle(frame, (int(right_eye_startx), int(right_eye_starty)), (int(right_eye_startx + right_eye_length), int(right_eye_starty + right_eye_width)), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break