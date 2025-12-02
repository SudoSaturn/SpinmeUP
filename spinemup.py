import argparse
import time
from pathlib import Path
import json
import subprocess
import re

from PIL import Image
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}

def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def detect_orientation_with_ollama(src_path: Path, model: str = "rotator") -> int:
    format_schema = '{"type":"object","properties":{"rotation":{"type":"string","enum":["0","90","180","270"]}},"required":["rotation"]}'
    try:
        result = subprocess.run(
            ["ollama", "run", model, "--format", format_schema,
             "Determine how many degrees it must be rotated CLOCKWISE so that people, objects, or scenery appear upright. If the image is already upright, return 0.",
             str(src_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            cleaned = re.sub(r"^Thinking\.+\s*", "", result.stdout.strip(), flags=re.IGNORECASE)
            data = json.loads(cleaned)
            rotation = int(data.get("rotation", "0"))
            if rotation in (0, 90, 180, 270):
                return rotation
    except Exception:
        pass
    return 0


def process_image(src_path: Path, input_root: Path, output_root: Path) -> None:
    try:
        rel = src_path.relative_to(input_root)
    except ValueError:
        return
    dst_path = output_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    rotation = detect_orientation_with_ollama(src_path)
    with Image.open(src_path) as img:
        if rotation in (90, 180, 270):
            img = img.rotate(rotation, expand=True)
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
    observer.schedule(ImageEventHandler(args.input, args.output), str(args.input), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
