# Get dataset, apply preprocessing and load pre-trained model. 

# general imports
import tarfile
import os, argparse, urllib
import cv2 as cv
import numpy as np
from pathlib import Path
import gdown

# pytorch imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()
    return args

def load_mobilenetv2():
    model = models.mobilenet_v2()
    num_classes = 2  # Assuming 2 classes for the eye state detection
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load("model/mobilenetv2_eye_state_detection.pt"))
    return model

### pass quantization as True from quantize_model.py to get calibration data. 
### Changing its value will affect the creating and usage of other dataloaders
### meant for train, val and test
def prepare_dataset(dataset_path, quantization: bool=False):

    # preprocessing
    train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.GaussianBlur(1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(dataset_path)
    classes = full_dataset.classes

    train_size = int(0.60 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = int(0.10 * len(full_dataset))
    ipu_test_size = int(0.10 * len(full_dataset))
    quantize_size = int(0.05 * len(full_dataset))

    train_dataset, val_dataset, test_dataset, ipu_test_dataset, quantize_dataset = random_split(
        full_dataset, [train_size, val_size, test_size, ipu_test_size, quantize_size])
    
    ipu_test_dataset.dataset.transform = basic_transform
    quantize_dataset.dataset.transform = basic_transform
    test_dataset.dataset.transform = basic_transform

    ipu_test_loader = DataLoader(ipu_test_dataset, batch_size=1, shuffle=False, num_workers=4)
    quantize_loader = DataLoader(quantize_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    if (quantization):
        return quantize_loader

    if get_args().train:
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = basic_transform

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

        return classes, train_loader, val_loader, test_loader, ipu_test_loader, quantize_loader

    return classes, test_loader, ipu_test_loader, quantize_loader

def train_model(num_epochs, train_loader, val_loader, criterion):

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

    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4) # stochastic gradient descent
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=0.3, gamma=0.1)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
                
                optimizer.zero_grad()   # clear gradient of all optimized tensors

                with torch.set_grad_enabled(phase=='train'): # only need gradients for training
                    outputs = model(inputs) # forward pass
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # compute gradient and update model params only during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    print('Training complete')
    model.to("cpu")
    torch.save(model.state_dict(), "model/mobilenetv2_eye_state_detection.pt")

def test_model(model, test_loader, criterion, classes, num_imgs):
    print("****************************")
    print("Testing Fine-Tuned Model...")

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    ground_truth = []
    prediction = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)

            ground_truth.extend(labels.cpu().numpy())
            prediction.extend(preds.cpu().numpy())

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects.double() / len(test_loader.dataset)

    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

    for i in range(num_imgs):
        print(f'Actual: {classes[ground_truth[i]]}, Predicted: {classes[prediction[i]]}')

    print("To view predictions of entire test set, please change num_imgs parameter in main()")

def export_to_onnx(model, models_dir):
    # prep for onnx export
    random_inputs = torch.randn(1, 3, 224, 224) # batch size, channels, height, width

    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    tmp_model_path = str(models_dir+"/mobilenetv2_eye_state_detection.onnx")
    torch.onnx.export(
        model,
        random_inputs,
        tmp_model_path,
        export_params=True,
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    print("Model exported to ONNX format!")

def main():
    args = get_args()

    dataset_path = "data/OACE"

    # get dataset from google drive
    # data_download_path = "data/OACE.tar.gz"
    # download_url = "https://drive.google.com/uc?export=download&id=1FaG8S0kLckhyTzYEUIl0E0Hzpry6G_xe"
    # gdown.download(download_url, data_download_path, quiet=False)
    # with tarfile.open(data_download_path, "r:gz") as tar:
    #     tar.extractall(path="data")
    
    # load finetuned model
    model = load_mobilenetv2()  
    model = model.to("cpu")

    criterion = nn.CrossEntropyLoss() # to learn both classes and not just one. 

    if args.train:
        classes, train_loader, val_loader, test_loader, ipu_test_loader, quantize_loader = prepare_dataset(dataset_path)
        train_model(args.num_epochs, train_loader, val_loader, criterion)
    else:
        classes, test_loader, ipu_test_loader, quantize_loader = prepare_dataset(dataset_path)

    # test model and export to onnx
    test_model(model, test_loader, criterion, classes, 10)
    export_to_onnx(model, "model")

if __name__ == "__main__":
    main()
