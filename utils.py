### utility and helper functions

import cv2
from pathlib import Path
from torchvision import transforms

import onnx, onnxruntime

# onnx model startup with vitisai
def load_quantized_model(model_path, model_name='mobilenetv2'):

    onnx_model_path = model_path
    model = onnx.load(onnx_model_path)  
    cache_key = 'mn2cachekey'

    '''
    TODO: Add argument to run inference on cpu and ipu
    '''

    if model_name=='mobilenetv3':
        cache_key = 'mn3cachekey'

    providers = ['VitisAIExecutionProvider'] # run inference on ipu
    cache_dir = Path().resolve()
    print(f"Cache directory set to: {cache_dir}")
    provider_options = [{
        'config_file': 'vaip_config.json',
        'cacheDir': str(cache_dir),
        'cacheKey': cache_key
    }]

    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=providers,
                                        provider_options=provider_options)
    
    return session

# image processing for inference
def process_image(image):

    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return image_transform(image)

def extract_eye_regions(rgb_frame, face_landmarks):

    left_eye_length = 1.6 * (face_landmarks[39, 0] - face_landmarks[36, 0])
    left_eye_width = 3.5 * (face_landmarks[41, 1] - face_landmarks[37, 1])
    left_eye_startx = face_landmarks[36, 0] - 0.3 * left_eye_length
    left_eye_starty = face_landmarks[37, 1] - 0.3 * left_eye_width

    right_eye_length = 1.6 * (face_landmarks[45, 0] - face_landmarks[42, 0])
    right_eye_width = 3.5 * (face_landmarks[47, 1] - face_landmarks[43, 1])
    right_eye_startx = face_landmarks[42, 0] - 0.3 * right_eye_length
    right_eye_starty = face_landmarks[43, 1] - 0.3 * right_eye_width

    left_eye_img = rgb_frame[int(left_eye_starty):int(left_eye_starty + left_eye_width), int(left_eye_startx):int(left_eye_startx + left_eye_length)]
    right_eye_img = rgb_frame[int(right_eye_starty):int(right_eye_starty + right_eye_width), int(right_eye_startx):int(right_eye_startx + right_eye_length)]

    return left_eye_img, right_eye_img, (int(left_eye_startx), int(left_eye_starty), int(left_eye_length), int(left_eye_width)), (int(right_eye_startx), int(right_eye_starty), int(right_eye_length), int(right_eye_width))

def draw_bbox_with_label(frame, bbox, label):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = f"{label}"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
