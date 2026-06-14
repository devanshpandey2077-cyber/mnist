# MNIST Draw — Browser Digit Recognition

This repository contains:

- `train_mnist_tfjs.py` — TensorFlow/Keras script to train a CNN on MNIST and export a TensorFlow.js model.
- `requirements.txt` — required Python packages.
- `index.html` — single-page frontend with a drawing canvas.
- `app.js` — canvas preprocessing, TensorFlow.js inference, and UI updates.

## Setup

1. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Train and export the model:

```powershell
python train_mnist_tfjs.py --epochs 15 --batch_size 128 --out_dir model
```

This writes a TensorFlow.js model into the `model/` directory. After training, confirm that `model/model.json` exists.

## Frontend

The frontend expects the exported TF.js model in `model/model.json`.

### Serve locally

From the repository root:

```powershell
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## How it works

- The canvas is 280×280 with thick white strokes.
- The frontend crops the drawn digit, resizes it to 20×20, centers it by center-of-mass, and places it in a 28×28 frame.
- The processed image is normalized to `[0.0, 1.0]` and fed into the TF.js model.
- Predictions update on `pointerup` so inference occurs once the stroke completes.

## Notes

- The model is trained on official MNIST.
- The preprocessing pipeline closely matches MNIST's digit alignment and centering.
- If you want higher robustness, increase epochs, add more augmentation, or retrain with drawn-digit examples.
