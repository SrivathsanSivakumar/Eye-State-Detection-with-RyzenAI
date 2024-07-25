from prepare_model_and_data import quantize_loader as get_quantize_loader

# vitis ai imports
import onnx, vai_q_onnx
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader


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

def quantize(quantize_loader):
    ### check model dimensions, features and input reqs
    onnx_model_path = "model/mobilenetv2_eye_state_detection.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    input_model_path = "model/mobilenetv2_eye_state_detection.onnx"
    output_model_path = "model/mobilenetv2_eye_state_detection.qdq.U8S8.onnx"
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
    
def main():
    quantize_loader = get_quantize_loader()
    quantize(quantize_loader)