# SpinmeUP

This project provides a Python script that watches a folder for new images, uses an small local model to determine their correct upright orientation, rotates them, and moves them to an output folder.

## Features

- Watches an input directory for new images.
- Supports formats: JPG, JPEG, PNG, HEIC, HEIF, TIF, TIFF, WEBP.
- Uses an Ollama model (`rotator`) to decide the rotation (0, 90, 180, 270 degrees).

## Requirements

- Python 3.10+
- Python packages:
  - `pillow`
  - `watchdog`
- Ollama CLI installed and on `PATH`.
- The model `rotator` (or any other model that reliably returns a JSON object with a `rotation` field)

```bash
ollama pull rotator
```

Install Python dependencies with:

```bash
pip install pillow watchdog
```

## Usage

Run the script with absolute input and output directories:

```bash
python your_script.py -i /path/to/input -o /path/to/output
```

- `-i` / `--input`: directory to watch for images.
- `-o` / `--output`: directory where rotated images will be written.

The script will:

1. Process all existing supported images under the input directory into the output directory (keeping their names and structure).

2. Continue running and automatically process any new images added to the input directory.
