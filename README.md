---
title: Tomato Disease Detector Backend
emoji: 🐢
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

#  AgriTech — Tomato Disease Diagnostic Center

AI-powered tomato leaf disease detection using an **entropy-weighted ensemble** of EfficientNet-B0 and ResNet-50, served via FastAPI with a modern React frontend.

## Features

- **Multi-Model Ensemble** — Combines EfficientNet-B0 (~5.3M params) and ResNet-50 (~25.6M params) with entropy-weighted averaging. More confident models automatically receive higher influence.
- **5 Disease Classes** — Bacterial Spot, Early Blight, Late Blight, Septoria Leaf Spot, and Healthy.
- **Saliency Heatmaps** — Occlusion-sensitivity maps showing which image regions matter most to the model's decision.
- **Visual Similarity Index** — Finds and displays top-matching reference cases from a pre-computed gallery using CNN embeddings.
- **LLM Personalised Advisor** — Generates tailored, structured treatment advice based on the model's diagnosis and the user's growing context (powered by Llama 3.3 via Groq).
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
| Deployment | Hugging Face Spaces + Vercel | Dockerized FastAPI backend on Spaces, static frontend on Vercel |

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- ONNX model files (see [Model Setup](#model-setup))

### Backend
```bash
pip install fastapi uvicorn onnxruntime pillow numpy groq python-dotenv
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
Create a `.env` file in the root for the backend:
```env
# Required for the LLM Personalised Advisor feature
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Create `frontend/.env` for production:
```env
VITE_API_URL=https://your-backend-url.com
```

## Deployment

### Backend on Hugging Face Spaces

Create a new Hugging Face Space with **Docker** as the SDK, then push this repo to the Space remote. The Space reads the YAML block at the top of this README and runs the backend on port `7860`.

```bash
git lfs install
git lfs track "train-new/**/*.onnx"
git lfs track "train-new/**/*.onnx.data"
git lfs track "similarity/*.npz"
git add .gitattributes
git add --renormalize train-new/efficient-net/tomato_disease_efficientnet.onnx
git add --renormalize train-new/efficient-net/tomato_disease_efficientnet.onnx.data
git add --renormalize train-new/resnet50/tomato_disease_resnet50.onnx
git add --renormalize similarity/embeddings.npz
git add -f train-new/efficient-net/tomato_disease_efficientnet.onnx.data
git add -f train-new/resnet50/tomato_disease_resnet50.onnx.data
git commit -m "Prepare backend for Hugging Face Spaces"
git remote add space https://huggingface.co/spaces/tanish19/tomato-disease
git push space main
```

Set `GROQ_API_KEY` and optionally `GROQ_MODEL` as Space secrets.

### Frontend on Vercel

Deploy the `frontend/` app on Vercel and set:

```env
VITE_API_URL=https://tanish19-tomato-disease.hf.space
```

## Model Setup

The ONNX model weight files (`.onnx.data`) are tracked with Git LFS for Hugging Face Spaces. After cloning without LFS, download them and place in:

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

### Visual Similarity Index Setup
The `/similar` endpoint requires a pre-computed index. To generate this, run the provided script in Google Colab against your training dataset:
```bash
python similarity/build_index.py --dataset /path/to/dataset --onnx train-new/efficient-net/tomato_disease_efficientnet.onnx
```
Then download the generated `embeddings.npz`, `manifest.json`, and `thumbnails/` into the `similarity/` directory of this repo.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/models` | Model metadata |
| `POST` | `/predict?mode=ensemble\|efnet\|resnet` | Disease prediction |
| `POST` | `/heatmap?model=efnet\|resnet` | Saliency heatmap |
| `POST` | `/similar?top_k=3` | Find visually similar reference cases |
| `POST` | `/advisor` | Generate LLM personalised treatment advice |

## Training

Both models were trained on Google Colab (T4 GPU) using a `TransformSubset` wrapper that ensures proper augmentation. Training scripts are in `_training_ref/`.

| Model | Val Acc | Epochs | Key Features |
|-------|---------|--------|--------------|
| EfficientNet-B0 | 98.55% | 15 | Standard augmentation |
| ResNet-50 (v2) | 97.60% | 25 | Mixup, label smoothing, CosineAnnealing, stronger augmentation |

## Project Structure

See [`claude.md`](claude.md) for full architecture documentation.

## License

MIT
