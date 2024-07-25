# Use quantized model for real time inference for eye state classification

# general imports
import cv2, dlib
import numpy as np
from pathlib import Path

# pytorch imports
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

import onnx, onnxruntime

classes = ["Closed", "Open"]

transform_to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_quantized_model(model_path):

    onnx_model_path = model_path
    model = onnx.load(onnx_model_path)  

    '''
    TODO: Add argument to run inference on cpu and ipu
    '''

    providers = ['VitisAIExecutionProvider'] # run inference on ipu
    cache_dir = Path().resolve()
    print(f"Cache directory set to: {cache_dir}")
    provider_options = [{
        'config_file': 'vaip_config.json',
        'cacheDir': str(cache_dir),
        'cacheKey': 'modelcachekey'
    }]

    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=providers,
                                        provider_options=provider_options)
    
    return session

def start_webcam(session):

    cam_capture = cv2.VideoCapture(0)

    # face landmark detection init
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
                left_eye_length = 1.6 * (face_landmarks[39, 0] - face_landmarks[36, 0])
                left_eye_width = 3.5 * (face_landmarks[41, 1] - face_landmarks[37, 1])
                left_eye_startx = face_landmarks[36, 0] - 0.3 * left_eye_length
                left_eye_starty = face_landmarks[37, 1] - 0.3 * left_eye_width

                # Calculate dimensions and position for the right eye
                right_eye_length = 1.6 * (face_landmarks[45, 0] - face_landmarks[42, 0])
                right_eye_width = 3.5 * (face_landmarks[47, 1] - face_landmarks[43, 1])
                right_eye_startx = face_landmarks[42, 0] - 0.3 * right_eye_length
                right_eye_starty = face_landmarks[43, 1] - 0.3 * right_eye_width
                    
                left_eye_img = rgb_frame[int(left_eye_starty):int(left_eye_starty + left_eye_width), int(left_eye_startx):int(left_eye_startx + left_eye_length)]
                if left_eye_img.size != 0:
                    left_eye_img_resized = cv2.resize(left_eye_img, (224, 224))
                    left_eye_img_tensor = transform_to_tensor(left_eye_img_resized).unsqueeze(0)
                    le_input_data = left_eye_img_tensor.numpy()

                    with torch.no_grad():
                        left_eye_output = session.run(None, {'input': le_input_data})
                        left_eye_output = torch.tensor(left_eye_output[0])
                        left_eye_prob = torch.nn.functional.softmax(left_eye_output, dim=1)
                        left_eye_confidence, left_eye_pred = torch.max(left_eye_prob, 1)

                    cv2.rectangle(frame, (int(left_eye_startx), int(left_eye_starty)), (int(left_eye_startx + left_eye_length), int(left_eye_starty + left_eye_width)), (0, 255, 0), 2)
                    # cv2.putText(frame, f"Left Eye State:{classes[left_eye_pred.item()]}, Confidence:{left_eye_confidence.item():.2f}", (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"{classes[left_eye_pred.item()]}", (int(left_eye_startx), int(left_eye_starty - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Process and save right eye region
                right_eye_img = rgb_frame[int(right_eye_starty):int(right_eye_starty + right_eye_width), int(right_eye_startx):int(right_eye_startx + right_eye_length)]
                if right_eye_img.size != 0:
                    right_eye_img_resized = cv2.resize(right_eye_img, (224, 224))
                    right_eye_tensor = transform_to_tensor(right_eye_img_resized).unsqueeze(0)
                    re_input_data = right_eye_tensor.numpy()

                    with torch.no_grad():
                        right_eye_output = session.run(None, {'input': re_input_data})
                        right_eye_output = torch.tensor(right_eye_output[0])
                        right_eye_prob = torch.nn.functional.softmax(right_eye_output, dim=1)
                        right_eye_confidence, right_eye_pred = torch.max(right_eye_prob, 1)

                    cv2.rectangle(frame, (int(right_eye_startx), int(right_eye_starty)), (int(right_eye_startx + right_eye_length), int(right_eye_starty + right_eye_width)), (0, 255, 0), 2)
                    # cv2.putText(frame, f"Right Eye State:{classes[right_eye_pred.item()]}, Confidence:{right_eye_confidence.item():.2f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"{classes[right_eye_pred.item()]}", (int(right_eye_startx), int(right_eye_starty - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    model_path = "model/mobilenetv2_eye_state_detection.qdq.U8S8.onnx"
    session = load_quantized_model(model_path)

    start_webcam(session)

if __name__ == "__main__":
    main()