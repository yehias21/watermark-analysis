import os

import onnxruntime as ort

# Module-level constants
WATERMARKS_DIR = "watermarks"
ONNX_MODEL_NAMES = ("rivagan_encoder.onnx", "rivagan_decoder.onnx", "stega_stamp.onnx")


def inspect_onnx_operations(model_path: str) -> None:
    """Inspect ONNX model operations.

    Args:
        model_path: Path to ONNX model file.
    """
    session = ort.InferenceSession(model_path)

    print("\nInputs:")
    for input in session.get_inputs():
        print(f"- Name: {input.name}, Shape: {input.shape}, Type: {input.type}")

    print("\nOutputs:")
    for output in session.get_outputs():
        print(f"- Name: {output.name}, Shape: {output.shape}, Type: {output.type}")

    print("\nProviders:", session.get_providers())


if __name__ == "__main__":
    for model_name in ONNX_MODEL_NAMES:
        model_path = os.path.join(WATERMARKS_DIR, model_name)
        print("Getting model information...")
        inspect_onnx_operations(model_path)
