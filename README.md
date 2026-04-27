#  AgriTech — Tomato Disease Diagnostic Center

AI-powered tomato leaf disease detection using an **entropy-weighted ensemble** of EfficientNet-B0 and ResNet-50, served via FastAPI with a modern React frontend.

## Features

- **Multi-Model Ensemble** — Combines EfficientNet-B0 (~5.3M params) and ResNet-50 (~25.6M params) with entropy-weighted averaging. More confident models automatically receive higher influence.
- **5 Disease Classes** — Bacterial Spot, Early Blight, Late Blight, Septoria Leaf Spot, and Healthy.
- **Saliency Heatmaps** — Occlusion-sensitivity maps showing which image regions matter most to the model's decision.
- **Clinical Reports** — Symptoms, treatment protocols, prevention strategies, and precautions for each disease.
- **Prediction History** — Local storage of past analyses for quick reference.
- **Confidence Thresholding** — Predictions below 60% confidence are flagged as uncertain.

## Architecture

```
Frontend (React + Vite)  →  FastAPI Backend  →  ONNX Runtime (CPU)
                                                  ├── EfficientNet-B0
                                                  └── ResNet-50
```

| Component | Tech | Details |
|-----------|------|---------|
| Frontend | React 19 + Vite 8 | Light theme, vanilla CSS, Outfit + Inter fonts |
| Backend | FastAPI + ONNX Runtime | Ensemble inference, heatmap generation |
| Models | EfficientNet-B0, ResNet-50 | ImageNet pretrained, fine-tuned on 13,829 tomato leaf images |
| Deployment | Vercel | Serverless Python function + static frontend |

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- ONNX model files (see [Model Setup](#model-setup))

### Backend
```bash
pip install fastapi uvicorn onnxruntime pillow numpy
python master_api.py
# → http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Environment Variables
Create `frontend/.env` for production:
```env
VITE_API_URL=https://your-backend-url.com
```

## Model Setup

The ONNX model weight files (`.onnx.data`) are too large for Git. After cloning, download them and place in:

```
train-new/
├── efficient-net/
│   ├── tomato_disease_efficientnet.onnx        ← in repo
│   ├── tomato_disease_efficientnet.onnx.data   ← download (~16MB)
│   └── class_mapping.json                      ← in repo
└── resnet50/
    ├── tomato_disease_resnet50.onnx            ← in repo
    ├── tomato_disease_resnet50.onnx.data       ← download (~94MB)
    └── class_mapping_resnet.json               ← in repo
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/models` | Model metadata |
| `POST` | `/predict?mode=ensemble\|efnet\|resnet` | Disease prediction |
| `POST` | `/heatmap?model=efnet\|resnet` | Saliency heatmap |

## Training

Both models were trained on Google Colab (T4 GPU) using a `TransformSubset` wrapper that ensures proper augmentation. Training scripts are in `_training_ref/`.

| Model | Val Acc | Epochs | Key Features |
|-------|---------|--------|--------------|
| EfficientNet-B0 | 98.55% | 15 | Standard augmentation |
| ResNet-50 (v2) | — | 25 | Mixup, label smoothing, CosineAnnealing, stronger augmentation |

## Project Structure

See [`claude.md`](claude.md) for full architecture documentation.

## License

MIT
