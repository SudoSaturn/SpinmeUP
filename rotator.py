import argparse
import time
import os
import torch
import onnxruntime
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image, ImageOps
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_SAVE_DIR = "model"
NUM_CLASSES = 4
IMAGE_SIZE = 384
ROTATIONS = {0: 0, 1: 90, 2: 180, 3: 270}

def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS

def load_image_safely(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGB", "L"):
        return img.convert("RGB")

    rgba_img = img.convert("RGBA")
    background = Image.new("RGB", rgba_img.size, (255, 255, 255))
    background.paste(rgba_img, mask=rgba_img)

    return background

def detect_orientation_with_onnx(src_path: Path) -> int:
    model_path = os.path.join(MODEL_SAVE_DIR, f"rotator.onnx")
    
    if not os.path.exists(model_path):
        print(f"Error: ONNX model file not found at {model_path}.")
        return 0

    image_transforms = T.Compose(
        [
            T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    PREFERRED_PROVIDERS = [
        "CUDAExecutionProvider",
        "MpsExecutionProvider",
        "ROCmExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    try:
        available_providers = onnxruntime.get_available_providers()

        chosen_provider = None
        for provider in PREFERRED_PROVIDERS:
            if provider in available_providers:
                chosen_provider = provider
                break

        if not chosen_provider:
            chosen_provider = "CPUExecutionProvider"

        ort_session = onnxruntime.InferenceSession(
            model_path, providers=[chosen_provider]
        )

    except Exception as e:
        print(f"Error loading ONNX model {model_path}: {e}")
        return 0

    try:
        image = load_image_safely(str(src_path))
    except FileNotFoundError:
        print(f"File not found: {src_path}")
        return 0
    except Exception as e:
        print(f"Error opening image {src_path}: {e}")
        return 0

    input_tensor = image_transforms(image).unsqueeze(0).cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    output = torch.from_numpy(ort_outs[0])
    _, predicted_idx = torch.max(output, 1)

    predicted_class = predicted_idx.item()
    rotation_degrees = ROTATIONS[predicted_class]
    return rotation_degrees

def process_image(src_path: Path, input_root: Path, output_root: Path) -> None:
    try:
        rel = src_path.relative_to(input_root)
    except ValueError:
        return
    dst_path = output_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    rotation_applied = detect_orientation_with_onnx(src_path)
    
    # To correct the image, rotate it in the opposite direction
    correction_rotation = (360 - rotation_applied) % 360
    
    with Image.open(src_path) as img:
        if correction_rotation in (90, 180, 270):
            img = img.rotate(correction_rotation, expand=True)
        img.save(dst_path)
    try:
        src_path.unlink()
    except OSError:
        pass

def process_all_existing(input_dir: Path, output_root: Path) -> None:
    for path in input_dir.rglob("*"):
        if is_supported_image(path):
            try:
                rel = path.relative_to(input_dir)
            except ValueError:
                continue
            if not (output_root / rel).exists():
                process_image(path, input_dir, output_root)

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, input_dir: Path, output_dir: Path):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if is_supported_image(path):
            try:
                rel = path.relative_to(self.input_dir)
            except ValueError:
                return
            if not (self.output_dir / rel).exists():
                process_image(path, self.input_dir, self.output_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    process_all_existing(args.input, args.output)
    observer = Observer()
    input_abs = args.input.resolve()
    output_abs = args.output.resolve()
    observer.schedule(ImageEventHandler(input_abs, output_abs), str(input_abs), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()