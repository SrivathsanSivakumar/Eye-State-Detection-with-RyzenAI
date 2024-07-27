from prepare_model_and_data import prepare_dataset
import argparse

# vitis ai imports
import onnx, vai_q_onnx
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default='mobilenetv2')
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
    onnx_model_path = f"model/{model_name}/{model_name}_eye_state_detection.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    input_model_path = f"model/{model_name}/{model_name}_eye_state_detection.onnx"
    output_model_path = f"model/{model_name}/{model_name}_eye_state_detection.qdq.U8S8.onnx"
    data_reader = mobilenet_calibartion_reader(quantize_loader)

    vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        data_reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=QuantType.QUInt8,
        # weight_type=QuantType.QInt8,
        enable_ipu_cnn=True,
        extra_options={'ActivationSymmetric': True}
    )

    print(f"Quantized Model Saved at {output_model_path}")

def quantized_model():
    pass

def main():
    args = get_args()
    quantize_loader = prepare_dataset("data/OACE", quantization=True)
    quantize(quantize_loader, args.model)

if __name__ == "__main__":
    main()