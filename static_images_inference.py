# perform inference using static images.

# General imports
from pathlib import Path
import argparse
import cv2
import dlib
from PIL import Image
import numpy as np
import utils

# PyTorch imports
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN

# ONNX imports
import onnx
import onnxruntime

classes = ["Closed", "Open"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default='mobilenetv2')
    parser.add_argument("--image", type=str)
    args = parser.parse_args()
    return args

def image_inference(image_paths, session):

    detector = MTCNN(select_largest=False, post_process=False)
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = dlib.full_object_detections()

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, probs = detector.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                box = box.astype(int)
                det = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                face_landmarks = [(item.x, item.y) for item in sp(rgb_frame, det).parts()]
                face_landmarks = np.array(face_landmarks)

                left_eye_img, right_eye_img, left_bbox, right_bbox = utils.extract_eye_regions(rgb_frame, face_landmarks)

                if left_eye_img.size != 0:
                    left_eye_img_resized = cv2.resize(left_eye_img, (224, 224))
                    left_eye_img_tensor = utils.process_image(left_eye_img_resized).unsqueeze(0)
                    le_input_data = left_eye_img_tensor.numpy()

                    with torch.no_grad():
                        left_eye_output = session.run(None, {'input': le_input_data})
                        left_eye_output = torch.tensor(left_eye_output[0])
                        left_eye_prob = torch.nn.functional.softmax(left_eye_output, dim=1)
                        left_eye_confidence, left_eye_pred = torch.max(left_eye_prob, 1)

                    utils.draw_bbox_with_label(frame, left_bbox, classes[left_eye_pred.item()], left_eye_confidence.item())

                if right_eye_img.size != 0:
                    right_eye_img_resized = cv2.resize(right_eye_img, (224, 224))
                    right_eye_img_tensor = utils.process_image(right_eye_img_resized).unsqueeze(0)
                    re_input_data = right_eye_img_tensor.numpy()

                    with torch.no_grad():
                        right_eye_output = session.run(None, {'input': re_input_data})
                        right_eye_output = torch.tensor(right_eye_output[0])
                        right_eye_prob = torch.nn.functional.softmax(right_eye_output, dim=1)
                        right_eye_confidence, right_eye_pred = torch.max(right_eye_prob, 1)

                    utils.draw_bbox_with_label(frame, right_bbox, classes[right_eye_pred.item()], right_eye_confidence.item())

        cv2.imshow('Image', frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def main():
    args = get_args()
    session = utils.load_quantized_model(f"model/{args.model}_eye_state_detection.qdq.U8S8.onnx", args.model)

    if args.image:
        image_inference([args.image], session)
    else:
        image_folder = "images"
        image_paths = list(Path(image_folder).glob('*.png'))
        image_inference(image_folder, session)

if __name__ == "__main__":
    main()
    