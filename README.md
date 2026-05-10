# AgriTech Tomato Disease Diagnostic Center

AI-powered tomato leaf disease detection using an entropy-weighted ensemble of EfficientNet-B0 and ResNet-50, served by FastAPI with a React/Vite frontend.

## Features

- Multi-model ensemble with entropy-weighted averaging
- Five tomato classes: Bacterial Spot, Early Blight, Late Blight, Septoria Leaf Spot, and Healthy
- Saliency heatmaps using occlusion sensitivity
- Similar reference cases from a pre-computed visual index
- Personalized advisor endpoint with Groq/Llama support and local fallback advice
- Clinical report with symptoms, treatment, prevention, and precautions
- Confidence thresholding for uncertain predictions

## Architecture

```text
React + Vite frontend -> FastAPI backend -> ONNX Runtime CPU
                                           -> EfficientNet-B0
                                           -> ResNet-50
```

| Component | Tech | Details |
| --- | --- | --- |
| Frontend | React 19 + Vite 8 | Vanilla CSS, Outfit + Inter fonts |
| Backend | FastAPI + ONNX Runtime | Prediction, heatmap, similarity, advisor APIs |
| Models | EfficientNet-B0, ResNet-50 | Fine-tuned on 13,829 tomato leaf images |
| Deployment | Vercel + Hugging Face Spaces | Vercel hosts frontend, HF Spaces hosts backend |

## Quick Start

### Backend

```bash
pip install -r requirements.txt
python master_api.py
```

Backend runs at:

```text
http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

## Environment Variables

Backend:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Frontend:

```env
VITE_API_URL=http://localhost:8000
```

Production frontend:

```env
VITE_API_URL=https://tanish19-tomato-disease.hf.space
```

## Deployment

The deployed app is split into two services:

- Frontend on Vercel
- Backend on Hugging Face Spaces at `https://tanish19-tomato-disease.hf.space`

For Vercel, deploy the `frontend/` directory only:

```text
Root Directory: frontend
Framework Preset: Vite
Build Command: npm run build
Output Directory: dist
```

For Hugging Face Spaces, use Docker. The backend Dockerfile runs:

```bash
uvicorn master_api:app --host 0.0.0.0 --port 7860
```

## Model Files

Binary model artifacts are tracked with Git LFS:

```text
train-new/efficient-net/tomato_disease_efficientnet.onnx
train-new/efficient-net/tomato_disease_efficientnet.onnx.data
train-new/resnet50/tomato_disease_resnet50.onnx
train-new/resnet50/tomato_disease_resnet50.onnx.data
similarity/embeddings.npz
```

After cloning, install Git LFS and pull the artifacts:

```bash
git lfs install
git lfs pull
```

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Health check |
| GET | `/models` | Model metadata |
| POST | `/predict?mode=ensemble\|efnet\|resnet` | Disease prediction |
| POST | `/heatmap?model=efnet\|resnet` | Occlusion heatmap |
| POST | `/similar?top_k=3` | Similar reference cases |
| POST | `/advisor` | Personalized treatment advice |

## Training Summary

Both models were trained on Google Colab with a fixed `TransformSubset` wrapper so train, validation, and test splits each use independent transforms.

| Model | Val Acc | Epochs | Notes |
| --- | --- | --- | --- |
| EfficientNet-B0 | 98.55% | 15 | Standard augmentation |
| ResNet-50 v2 | 97.60% | 25 | Mixup, label smoothing, cosine annealing, stronger augmentation |

## Project Structure

See [claude.md](claude.md) for the full architecture notes.

## License

MIT
