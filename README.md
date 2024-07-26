# Eye State Classification using RyzenAI with MobileNetV2, ResNet50 and A Custom CNN

## Introduction

A MobileNetV2 model is adapted to the classification task of eye state classification i.e. open or closed eyes. Each eye is individually recognized and labelled. The model is converted and quantized to be compatible with AMD's Ryzen AI chip. 

## Hardware Requirements

1. AMD Ryzen AI Chip/Hardware
2. Follow setup for IPU [here](https://ryzenai.docs.amd.com/en/latest/inst.html)

## Steps to Run Inference

1. Clone the repository

    `git clone https://github.com/SrivathsanSivakumar/Eye-State-Detection-with-RyzenAI`

2. Open Anaconda Prompt and move into the repository. Make sure to activate your conda env!
3. Run the following command to install the necessary packages

    `pip install requirements.txt`
4. Run the following command for __real-time__ webcam inference

    `python webcam_inference.py`

    This uses the onnx model quantized and converted to run with AMD's Ryzen AI chip. To explore more options go [here](#steps-to-run-full-project) 

    *If you do not have access to a webcam you can run inference using static images with the command*

    *`python static_images_inference.py`*

    *To test with a custom image use command*

    *`python static_images_inference.py --custom <your image path>`*

## Steps to Run Full Project

## Model Performance Comparison
