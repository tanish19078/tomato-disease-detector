import io
import json
import base64
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import os
from contextlib import asynccontextmanager

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Disease Knowledge Base ─────────────────────────────────────────────
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "label": "Bacterial Spot",
        "severity": "Moderate",
        "symptoms": [
            "Small, dark, water-soaked spots on leaves and fruit",
            "Leaf yellowing and premature drop",
            "Raised, scab-like lesions on fruit surface"
        ],
        "treatment": [
            "Apply copper-based bactericides early in the season",
            "Avoid overhead irrigation — use drip systems",
            "Remove and destroy infected plant debris immediately",
            "Practice crop rotation with non-solanaceous crops"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Sanitize tools between plants",
            "Ensure adequate plant spacing for air circulation"
        ],
        "precautions": [
            "Wash hands thoroughly after handling infected plants",
            "Do not touch healthy plants after touching spotted leaves"
        ]
    },
    "Tomato___Early_blight": {
        "label": "Early Blight",
        "severity": "Moderate-High",
        "symptoms": [
            "Dark concentric rings forming target-shaped spots",
            "Lower/older leaves affected first, progressing upward",
            "Yellowing tissue surrounding lesions",
            "Stem lesions near the soil line"
        ],
        "treatment": [
            "Apply chlorothalonil or mancozeb fungicide",
            "Remove infected lower leaves to slow spread",
            "Apply organic neem oil solution as preventive spray",
            "Improve air circulation through proper pruning"
        ],
        "prevention": [
            "Mulch around base to prevent soil splash",
            "3-year crop rotation away from Solanaceae",
            "Water at the base, never on foliage"
        ],
        "precautions": [
            "Disinfect pruning shears with 70% alcohol",
            "Avoid harvesting fruits during damp periods"
        ]
    },
    "Tomato___Late_blight": {
        "label": "Late Blight",
        "severity": "Critical",
        "symptoms": [
            "Large, dark brown water-soaked patches on leaves",
            "White-gray fungal growth on leaf undersides",
            "Rapid wilting and plant death in humid conditions",
            "Brown, firm rot on green fruit"
        ],
        "treatment": [
            "IMMEDIATE: Remove and destroy all infected plants — do NOT compost",
            "Apply systemic fungicide (metalaxyl/mefenoxam) to remaining plants",
            "Alert neighboring farms — this pathogen spreads via wind",
            "Harvest any unaffected fruit immediately"
        ],
        "prevention": [
            "Plant resistant cultivars (e.g., 'Defiant', 'Mountain Magic')",
            "Monitor weather — outbreaks follow cool, wet periods",
            "Avoid evening watering to reduce leaf wetness duration"
        ],
        "precautions": [
            "Securely bag infected plants before moving them off-site",
            "Immediately sterilize footwear post-exposure to infected soil"
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "label": "Septoria Leaf Spot",
        "severity": "Moderate",
        "symptoms": [
            "Numerous small circular spots (1-3mm) with dark borders",
            "Gray-white centers with tiny black fruiting bodies",
            "Lower leaves affected first, defoliation progresses upward",
            "Rarely affects fruit directly, but weakens the plant"
        ],
        "treatment": [
            "Apply copper fungicide or chlorothalonil at first sign",
            "Remove and destroy affected lower leaves",
            "Ensure good drainage to reduce soil moisture",
            "Apply organic compost tea as a biological control"
        ],
        "prevention": [
            "Mulch heavily to prevent rain splash from soil",
            "Staking/caging plants improves airflow",
            "Avoid working with wet plants to prevent spread"
        ],
        "precautions": [
            "Avoid handling foliage when it is wet",
            "Clean and sterilize stakes and cages at end of season"
        ]
    },
    "Tomato___healthy": {
        "label": "Healthy",
        "severity": "None",
        "symptoms": [
            "Vibrant green foliage with no visible spots or lesions",
            "Strong stem structure and normal leaf shape",
            "Healthy fruit development with no discoloration"
        ],
        "treatment": [
            "Continue regular watering schedule (1-2 inches/week)",
            "Maintain balanced N-P-K fertilization",
            "Monitor weekly for early signs of any disease"
        ],
        "prevention": [
            "Keep up current care regimen",
            "Rotate crops annually",
            "Apply preventive organic neem spray every 2 weeks"
        ],
        "precautions": [
            "Maintain vigilance and routinely inspect lower canopy",
            "Regularly monitor local agricultural extension for disease alerts"
        ]
    }
}

ALL_CLASSES = sorted(DISEASE_INFO.keys())
CONFIDENCE_THRESHOLD = 60.0  # Below this, flag prediction as uncertain

# ─── Model Registry ─────────────────────────────────────────────────────
MODEL_META = {
    "efnet": {
        "name": "EfficientNet-B0",
        "params": "~5.3M",
        "best_val_acc": 98.55,
        "size_mb": 16,
        "onnx_path": "train-new/efficient-net/tomato_disease_efficientnet.onnx",
        "mapping_path": "train-new/efficient-net/class_mapping.json",
    },
    "resnet": {
        "name": "ResNet-50",
        "params": "~25.6M",
        "best_val_acc": 98.63,
        "size_mb": 94,
        "onnx_path": "train-new/resnet50/tomato_disease_resnet50.onnx",
        "mapping_path": "train-new/resnet50/class_mapping_resnet.json",
    },
}

sessions: dict[str, ort.InferenceSession] = {}
mappings: dict[str, dict] = {}


def load_model(key: str, meta: dict) -> bool:
    """Load an ONNX model and its class mapping. Returns True on success."""
    onnx_path = meta["onnx_path"]
    mapping_path = meta["mapping_path"]

    if not os.path.exists(onnx_path):
        logger.warning(f"⚠ Skipping '{key}' — ONNX file not found: {onnx_path}")
        return False
    if not os.path.exists(mapping_path):
        logger.warning(f"⚠ Skipping '{key}' — mapping not found: {mapping_path}")
        return False

    try:
        logger.info(f"Loading '{meta['name']}' from {onnx_path}...")
        sessions[key] = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        with open(mapping_path, "r") as f:
            mappings[key] = json.load(f)
        logger.info(f"✓ '{meta['name']}' loaded — {len(mappings[key])} classes")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load '{key}': {e}")
        return False


# ─── App Lifecycle ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler (replaces deprecated on_event)."""
    # Startup
    for key, meta in MODEL_META.items():
        load_model(key, meta)
    logger.info(f"Startup complete. Active models: {list(sessions.keys())}")
    yield
    # Shutdown
    sessions.clear()
    mappings.clear()
    logger.info("Shutdown complete. Models unloaded.")


app = FastAPI(
    title="Tomato Diagnostic Center — Unified API",
    description="Multi-model inference engine supporting EfficientNet-B0 and ResNet-50 with ensemble mode.",
    version="2.1.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Image Preprocessing ────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Transform raw bytes into [1, 3, 224, 224] FP32 tensor with ImageNet normalization."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    return np.expand_dims(img_array, axis=0)  # [1, 3, 224, 224]


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def run_inference(key: str, input_tensor: np.ndarray) -> dict:
    """Run a single model and return structured result."""
    sess = sessions[key]
    mapping = mappings[key]

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    raw_logits = sess.run([output_name], {input_name: input_tensor})[0]
    probs = softmax(raw_logits)[0]

    idx = int(np.argmax(probs))
    class_name = mapping[str(idx)]
    confidence = round(float(probs[idx]) * 100, 2)

    distribution = {
        mapping[str(i)]: round(float(probs[i]) * 100, 2)
        for i in range(len(probs))
    }

    return {
        "prediction": class_name,
        "confidence": confidence,
        "distribution": distribution,
        "raw_probs": probs,  # kept for ensemble math, stripped before response
    }


def ensemble_predict(per_model: dict) -> tuple[str, float, bool]:
    """
    Entropy-weighted ensemble: models with sharper (more confident) probability
    distributions receive higher weight.  This prevents a confused model from
    dragging down a confident, correct model.

    Weight_i = 1 / (entropy_i + epsilon)   (normalized across models)
    """
    epsilon = 1e-8

    # 1. Build per-model aligned probability vectors and compute entropy weights
    model_vecs: list[np.ndarray] = []
    model_weights: list[float] = []

    for key, result in per_model.items():
        raw = result["raw_probs"]
        mapping = mappings.get(key)
        if not mapping:
            continue

        # Align to ALL_CLASSES order
        vec = np.zeros(len(ALL_CLASSES), dtype=np.float64)
        for i in range(len(raw)):
            cls = mapping[str(i)]
            cls_idx = ALL_CLASSES.index(cls)
            vec[cls_idx] = float(raw[i])

        model_vecs.append(vec)

        # Shannon entropy — lower = more confident
        p = np.clip(vec, epsilon, 1.0)
        entropy = -np.sum(p * np.log(p))
        model_weights.append(1.0 / (entropy + epsilon))

    # 2. Normalize weights to sum to 1
    total_w = sum(model_weights)
    model_weights = [w / total_w for w in model_weights]

    # 3. Weighted average
    avg_probs = np.zeros(len(ALL_CLASSES), dtype=np.float64)
    for vec, w in zip(model_vecs, model_weights):
        avg_probs += w * vec

    best_idx = int(np.argmax(avg_probs))
    final_class = ALL_CLASSES[best_idx]
    final_confidence = round(float(avg_probs[best_idx]) * 100, 2)
    models_agree = all(r["prediction"] == final_class for r in per_model.values())

    # Log weights for debugging
    weight_info = ", ".join(
        f"{k}={w:.2%}" for k, w in zip(per_model.keys(), model_weights)
    )
    logger.info(f"Ensemble weights: {weight_info}")

    return final_class, final_confidence, models_agree


# ─── Leaf Validation (OOD Detection) ────────────────────────────────────
def validate_leaf_image(img_bytes: bytes) -> dict:
    """
    Check whether an image likely contains a tomato leaf using colour
    histogram heuristics.  Works on the raw bytes before any model-specific
    preprocessing.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((100, 100))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total = float(r.size)

    # Green-dominant pixels (healthy leaf tissue)
    green_ratio = np.sum((g > r) & (g > b) & (g > 50)) / total

    # Brown/tan pixels (diseased tissue, stems)
    brown_ratio = np.sum((r > g * 0.85) & (g > b) & (r > 50) & (r < 220)) / total

    # Dark spot pixels (necrosis)
    dark_ratio = np.sum((r < 80) & (g < 80) & (b < 80) & ((r + g + b) > 30)) / total

    # Yellow pixels (chlorosis, early disease)
    yellow_ratio = np.sum((r > 150) & (g > 150) & (b < 100)) / total

    # Combined leaf-likeness score
    leaf_score = green_ratio + brown_ratio * 0.6 + dark_ratio * 0.3 + yellow_ratio * 0.5

    # Texture check (std dev across channels — solid colours fail)
    channel_std = float(np.mean([np.std(r), np.std(g), np.std(b)]))
    has_texture = channel_std > 20

    threshold = 0.20  # ≥20 % of pixels must look leaf-like
    is_leaf = leaf_score >= threshold and has_texture

    reason = ""
    if not has_texture:
        reason = "Image appears to be a solid colour or graphic, not a photograph of a leaf."
    elif leaf_score < threshold:
        reason = "Image does not appear to contain plant or leaf tissue."

    return {
        "is_leaf": is_leaf,
        "leaf_score": round(leaf_score * 100, 1),
        "reason": reason,
    }


# ─── API Endpoints ──────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "models_loaded": list(sessions.keys()),
        "total_classes": len(ALL_CLASSES),
        "api": "Tomato Diagnostic Center v2.1",
    }


@app.get("/models")
def get_models():
    """Return available models and their metadata."""
    return {
        key: {k: v for k, v in meta.items() if k not in ("onnx_path", "mapping_path")}
        for key, meta in MODEL_META.items()
        if key in sessions
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Query("ensemble", enum=["efnet", "resnet", "ensemble"]),
):
    """Run disease prediction on an uploaded tomato leaf image."""
    # Validate state
    if not sessions:
        raise HTTPException(status_code=503, detail="No models are loaded.")

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: JPEG, PNG, WebP.",
        )

    # Validate file size (max 10MB)
    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")

    # Validate that the image actually looks like a leaf
    validation = validate_leaf_image(img_bytes)

    try:
        input_tensor = preprocess_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    # Run inference on requested models
    per_model = {}
    target_models = list(sessions.keys()) if mode == "ensemble" else [mode]

    for key in target_models:
        if key not in sessions:
            raise HTTPException(status_code=400, detail=f"Model '{mode}' is not loaded.")
        per_model[key] = run_inference(key, input_tensor)

    # Determine final prediction
    if mode == "ensemble" and len(per_model) > 1:
        final_class, final_confidence, models_agree = ensemble_predict(per_model)
    else:
        active_key = list(per_model.keys())[0]
        final_class = per_model[active_key]["prediction"]
        final_confidence = per_model[active_key]["confidence"]
        models_agree = True

    info = DISEASE_INFO.get(final_class, {})

    # Strip internal data before response
    diagnostics = {}
    for key, result in per_model.items():
        diagnostics[key] = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "distribution": result["distribution"],
        }

    # Build response
    response = {
        "success": True,
        "prediction": info.get("label", final_class),
        "prediction_class": final_class,
        "confidence": final_confidence,
        "is_confident": final_confidence >= CONFIDENCE_THRESHOLD,
        "severity": info.get("severity", "Unknown"),
        "symptoms": info.get("symptoms", []),
        "treatment": info.get("treatment", []),
        "prevention": info.get("prevention", []),
        "precautions": info.get("precautions", []),
        "models_agree": models_agree,
        "mode": mode,
        "diagnostics": diagnostics,
        "validation": validation,
    }

    # Log prediction summary
    logger.info(
        f"Prediction: {final_class} ({final_confidence}%) | "
        f"Mode: {mode} | Agree: {models_agree} | Confident: {response['is_confident']}"
    )

    return JSONResponse(response)


@app.post("/heatmap")
async def generate_heatmap(
    file: UploadFile = File(...),
    model: str = Query("efnet", enum=["efnet", "resnet"]),
):
    """Generate an occlusion-sensitivity heatmap showing which image regions matter most."""
    if model not in sessions:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not loaded.")

    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_normalized = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # Baseline inference
        baseline_input = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0).astype(np.float32)
        sess = sessions[model]
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        baseline_probs = softmax(sess.run([output_name], {input_name: baseline_input})[0])[0]
        predicted_class = int(np.argmax(baseline_probs))
        baseline_conf = float(baseline_probs[predicted_class])

        # Occlusion sensitivity — 7×7 grid
        grid = 7
        patch = 224 // grid
        importance = np.zeros((grid, grid), dtype=np.float32)

        for i in range(grid):
            for j in range(grid):
                occluded = img_normalized.copy()
                y1, y2 = i * patch, min((i + 1) * patch, 224)
                x1, x2 = j * patch, min((j + 1) * patch, 224)
                occluded[y1:y2, x1:x2, :] = 0.0  # Zero in normalized space = ImageNet mean
                occ_input = np.expand_dims(np.transpose(occluded, (2, 0, 1)), axis=0).astype(np.float32)
                occ_probs = softmax(sess.run([output_name], {input_name: occ_input})[0])[0]
                importance[i, j] = max(0.0, baseline_conf - float(occ_probs[predicted_class]))

        # Normalize to [0, 1]
        if importance.max() > 0:
            importance /= importance.max()

        # Upscale to 224×224 with bilinear interpolation
        heatmap_small = Image.fromarray((importance * 255).astype(np.uint8), mode='L')
        heatmap_img = heatmap_small.resize((224, 224), Image.BILINEAR)
        v = np.array(heatmap_img, dtype=np.float32) / 255.0

        # Build RGBA overlay: transparent → yellow → orange → red
        rgba = np.zeros((224, 224, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.where(v > 0.1, 255, 0).astype(np.uint8)
        rgba[:, :, 1] = np.where(v > 0.1, ((1 - v) * 180).astype(np.uint8), 0)
        rgba[:, :, 2] = 0
        rgba[:, :, 3] = np.where(v > 0.1, (v * 180).astype(np.uint8), 0)

        buffer = io.BytesIO()
        Image.fromarray(rgba, mode='RGBA').save(buffer, format='PNG')
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        mapping = mappings[model]
        logger.info(f"Heatmap generated for {mapping[str(predicted_class)]} via {model}")

        return JSONResponse({
            "success": True,
            "heatmap_base64": heatmap_b64,
            "model_used": model,
            "predicted_class": mapping[str(predicted_class)],
            "baseline_confidence": round(baseline_conf * 100, 2),
        })
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
