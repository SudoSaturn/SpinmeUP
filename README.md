# SpinMeUP - Automatic Image Orientation Correction

A Python script that monitors an input directory for new images and automatically corrects their orientation using an ONNX model.

## Features

- **Real-time monitoring**: Uses watchdog to detect new images in input directory
- **ONNX model inference**: Fast orientation detection using pre-trained model
- **Automatic correction**: Rotates images to make them upright
- **Batch processing**: Handles multiple images automatically

## Usage

```bash
python3 rotator.py --input /path/to/images --output /path/to/corrected
```

## Arguments

- `--input, -i`: Input directory to monitor (required)
- `--output, -o`: Output directory for corrected images (required)

## How It Works

1. **Model Loading**: Loads ONNX model from `model/orientation_model_v2_0.9882.onnx`
2. **Detection**: For each new image, the model predicts the rotation that was APPLIED
3. **Correction**: To correct the image, it rotates in the opposite direction:
   - If model predicts "90°" was applied → rotate by 270° to correct
   - If model predicts "270°" was applied → rotate by 90° to correct
   - If model predicts "180°" was applied → rotate by 180° to correct
4. **File Management**: Removes original image after successful correction

## Dependencies

- `torch` - Tensor operations
- `onnxruntime` - ONNX model inference
- `numpy` - Array operations  
- `torchvision` - Image transforms
- `pillow` - Image processing
- `watchdog` - File system monitoring

## Output

The script outputs rotation degrees for each processed image:
- `0` - Image is upright
- `90` - Image needs 90° correction
- `180` - Image needs 180° correction  
- `270` - Image needs 270° correction
