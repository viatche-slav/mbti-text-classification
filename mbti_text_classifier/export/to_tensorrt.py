import subprocess
from pathlib import Path


def export_tensorrt(cfg, onnx_path, output_path):
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    batch_size = cfg.batch_size
    max_length = cfg.data.tokenizer.max_length

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        "--minShapes=input_ids:1x1,attention_mask:1x1",
        f"--optShapes=input_ids:{batch_size}x{max_length},\
            attention_mask:{batch_size}x{max_length}",
        f"--maxShapes=input_ids:{batch_size*2}x{max_length},\
            attention_mask:{batch_size*2}x{max_length}",
        "--fp16",
        "--verbose",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TensorRT conversion failed: {e.stderr}")
