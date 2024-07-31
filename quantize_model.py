from prepare_model_and_data import prepare_dataset
import argparse
import torch
import utils

# vitis ai imports
import onnx, vai_q_onnx
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default='mobilenetv2')
    parser.add_argument("--test_only", action='store_true')
    args = parser.parse_args()
    return args

class MobileNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader):
        super().__init__()
        self.iterator = iter(data_loader)

    def get_next(self) -> dict:
        try:
            images, labels = next(self.iterator)
            return {"input": images.numpy()}
        except Exception:
            return None

def mobilenet_calibartion_reader(data_loader):
    return MobileNetCalibrationDataReader(data_loader)

def quantize(quantize_loader, model_name):
    ### check model dimensions, features and input reqs
    print(f"Quantizing {model_name}...")
    onnx_model_path = f"models/{model_name}_eye_state_detection.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    input_model_path = onnx_model_path
    output_model_path = f"models/{model_name}_eye_state_detection.qdq.U8S8.onnx"
    data_reader = mobilenet_calibartion_reader(quantize_loader)

    vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        data_reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        enable_ipu_cnn=True,
        # execution_providers=['CPUExecutionProvider'],
        extra_options={'ActivationSymmetric': True}
    )

    print(f"Quantized Model Saved at {output_model_path}")

def test_quantized_model(dataloader, session):
    print("\n****************************")
    print("Testing Quantized Model...\n")

    running_loss = 0.0
    running_corrects = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:

            # input processing for quantized model
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

            input_data = inputs.numpy()

            # inference
            outputs = session.run(None, {'input': input_data})
            outputs = torch.tensor(outputs[0])
            
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            preds = preds.clone().detach()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)

    total_loss = running_loss/len(dataloader.dataset)
    total_acc = running_corrects.double()/len(dataloader.dataset)

    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

def main():
    args = get_args()
    
    if not args.test_only:
        quantize_loader = prepare_dataset("data/OACE", quantization=True)
        quantize(quantize_loader, args.model)

    ipu_test_loader = prepare_dataset("data/OACE", ipu_test=True)
    session = utils.load_quantized_model(f"models/{args.model}_eye_state_detection.qdq.U8S8.onnx", args.model)
    test_quantized_model(ipu_test_loader, session)

if __name__ == "__main__":
    main()