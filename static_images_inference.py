# perform inference using static images.


# general imports
from prepare_model_and_data import prepare_dataset
from pathlib import Path

# pytorch imports
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN

import onnx, onnxruntime

classes = ["Closed", "Open"]

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

def inference_on_static_images(dataloader, session, num_imgs=10,
                               custom: bool=False, image_path: str=None):
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0

    ground_truth = []
    prediction = []

    if custom:
        pass
    else:
        with torch.no_grad():
            for inputs, labels in dataloader:

                # input processing for quantized model
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')

                input_data = inputs.numpy()

                # onnx inference
                outputs = session.run(None, {'input': input_data})
                outputs = torch.tensor(outputs[0])
                
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                preds = preds.clone().detach()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

                ground_truth.extend(labels.cpu().numpy())
                prediction.extend(preds.cpu().numpy())

        total_loss = running_loss/len(dataloader.dataset)
        total_acc = running_corrects.double()/len(dataloader.dataset)

        print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

        # Print ground truth and predicted labels
        for i in range(num_imgs):
            print(f'Actual: {classes[ground_truth[i]]}, Predicted: {classes[prediction[i]]}')

        print("To view predictions of entire test set, please change num_imgs parameter in main()")

def main():
    dataloader = prepare_dataset("data/OACE", static_inference=True)
    session = load_quantized_model("model/mobilenetv2/mobilenetv2_eye_state_detection.qdq.U8S8.onnx")
    
    inference_on_static_images(dataloader, session, num_imgs=10)

if __name__ == "__main__":
    main()
    