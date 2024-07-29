### utility and helper functions

import cv2
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models

import onnx, onnxruntime

### ***************************************************************************
###                         Inference Helper Functions
### ***************************************************************************

# onnx model startup with vitisai
def load_quantized_model(model_path, model_name):

    onnx_model_path = model_path
    model = onnx.load(onnx_model_path)  
    cache_key = 'modelcachekey' # default cache key - used for mobilennetv2
    
    '''
    TODO: Add argument to run inference on cpu and ipu
    '''

    if model_name == "mobilenetv3":
        cache_key = 'modelcachekey_mn3'


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

    left_eye_length = 2 * (face_landmarks[39, 0] - face_landmarks[36, 0])
    left_eye_width = 4 * (face_landmarks[41, 1] - face_landmarks[37, 1])
    left_eye_startx = face_landmarks[36, 0] - 0.3 * left_eye_length
    left_eye_starty = face_landmarks[37, 1] - 0.3 * left_eye_width

    right_eye_length = 2 * (face_landmarks[45, 0] - face_landmarks[42, 0])
    right_eye_width = 4 * (face_landmarks[47, 1] - face_landmarks[43, 1])
    right_eye_startx = face_landmarks[42, 0] - 0.3 * right_eye_length
    right_eye_starty = face_landmarks[43, 1] - 0.3 * right_eye_width

    left_eye_img = rgb_frame[int(left_eye_starty):int(left_eye_starty + left_eye_width), int(left_eye_startx):int(left_eye_startx + left_eye_length)]
    right_eye_img = rgb_frame[int(right_eye_starty):int(right_eye_starty + right_eye_width), int(right_eye_startx):int(right_eye_startx + right_eye_length)]

    return left_eye_img, right_eye_img, (int(left_eye_startx), int(left_eye_starty), int(left_eye_length), int(left_eye_width)), (int(right_eye_startx), int(right_eye_starty), int(right_eye_length), int(right_eye_width))

def draw_bbox_with_label(frame, bbox, label, confidence=None):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if confidence != None: text = f"{label} ({confidence:.2f})" 
    else: text = f"{label}"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


### ***************************************************************************
###                       Train and Test Helper Functions
### ***************************************************************************


def load_mobilenetv2():
    model = models.mobilenet_v2()
    num_classes = 2  # Assuming 2 classes for the eye state detection
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load("model/mobilenetv2_eye_state_detection.pt"))
    return model

def load_mobilenetv3():
    model = models.mobilenet_v3_large()
    num_classes = 2
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load("model/mobilenetv3_eye_state_detection.pt"))
    return model

def get_fresh_model(model_name):
    if model_name == "mobilenetv2":
        print("Initialized Fresh MobileNetV2 Model")
        model = models.mobilenet_v2(weights="IMAGENET1K_V2")

        # modify final layer to match the number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2) # open eyes and close eyes - 2 classes

        # freeze base layers of model
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad=True
            else:
                param.requires_grad=False
        
        return model
    
    elif model_name == "mobilenetv3":
        print("Initialized Fresh MobileNetV3 Model")
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")

        # modify final layer to match the number of classes
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 2) # open eyes and close eyes - 2 classes
        
        # freeze base layers of model
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad=True
            else:
                param.requires_grad=False

        return model

    else: pass # TODO: add support for resnet50