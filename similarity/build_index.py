"""
Similarity Index Builder
========================
Run this on Google Colab (or locally with the dataset) to build the
reference gallery for the /similar endpoint.

Usage (Colab):
    1. Upload this script or paste into a cell
    2. Mount Google Drive with the tomato dataset
    3. Set DATASET_DIR below to point to the extracted dataset
    4. Run the script
    5. Download the generated files (embeddings.npz, manifest.json, thumbnails/)
    6. Place them in the `similarity/` directory of the project

Usage (Local — if you have dataset + ONNX model):
    python similarity/build_index.py --dataset /path/to/dataset --onnx train-new/efficient-net/tomato_disease_efficientnet.onnx
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ─── Configuration ──────────────────────────────────────────────────────
CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]

MAX_PER_CLASS = 60        # Max images per class for the gallery
THUMBNAIL_SIZE = (120, 90)
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── Image Preprocessing (mirrors master_api.py) ───────────────────────
def preprocess_image(img_path: str) -> np.ndarray:
    """Preprocess a single image to [1, 3, 224, 224] tensor."""
    image = Image.open(img_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    return np.expand_dims(img_array, axis=0)  # [1, 3, 224, 224]


def create_thumbnail(img_path: str) -> Image.Image:
    """Create a small JPEG thumbnail."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
    # Pad to exact size if needed
    canvas = Image.new("RGB", THUMBNAIL_SIZE, (245, 247, 250))
    offset = ((THUMBNAIL_SIZE[0] - img.width) // 2, (THUMBNAIL_SIZE[1] - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def build_index(dataset_dir: str, onnx_path: str, output_dir: str):
    """Build the similarity index from a dataset directory."""
    import onnxruntime as ort

    try:
        import onnx
        from onnx import TensorProto, helper as onnx_helper
    except ImportError:
        print("ERROR: 'onnx' package required. Install with: pip install onnx")
        sys.exit(1)

    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "thumbnails").mkdir(exist_ok=True)

    # ─── Set up embedding extraction ─────────────────────────────────
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)

    # Find the penultimate layer (input to last Gemm/MatMul)
    emb_name = None
    for node in reversed(model.graph.node):
        if node.op_type in ("Gemm", "MatMul"):
            emb_name = node.input[0]
            break

    if not emb_name:
        print("ERROR: Could not find embedding node in ONNX graph")
        sys.exit(1)

    # Add embedding as output
    emb_output = onnx_helper.make_tensor_value_info(emb_name, TensorProto.FLOAT, None)
    model.graph.output.append(emb_output)

    model_bytes = model.SerializeToString()
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"✓ Embedding session ready — node: '{emb_name}'")

    # ─── Collect images ──────────────────────────────────────────────
    embeddings = []
    manifest = []
    global_idx = 0

    for cls in CLASSES:
        cls_dir = dataset_path / cls
        if not cls_dir.exists():
            print(f"⚠ Class directory not found: {cls_dir}")
            continue

        # Get all image files
        img_files = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ])

        # Sample subset
        if len(img_files) > MAX_PER_CLASS:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(img_files), MAX_PER_CLASS, replace=False)
            img_files = [img_files[i] for i in sorted(indices)]

        print(f"  {cls}: {len(img_files)} images")

        for img_path in img_files:
            try:
                # Extract embedding
                tensor = preprocess_image(str(img_path))
                outputs = session.run([emb_name], {input_name: tensor})
                emb = outputs[0].flatten().astype(np.float32)
                embeddings.append(emb)

                # Create thumbnail
                thumb = create_thumbnail(str(img_path))
                thumb.save(output_path / "thumbnails" / f"{global_idx}.jpg", "JPEG", quality=75)

                # Add to manifest
                manifest.append({
                    "index": global_idx,
                    "class": cls,
                    "source_file": img_path.name,
                })

                global_idx += 1
            except Exception as e:
                print(f"  ⚠ Failed: {img_path.name} — {e}")

    # ─── Save index ──────────────────────────────────────────────────
    if not embeddings:
        print("ERROR: No embeddings extracted!")
        sys.exit(1)

    emb_matrix = np.stack(embeddings, axis=0)  # [N, 1280]
    np.savez_compressed(
        str(output_path / "embeddings.npz"),
        embeddings=emb_matrix,
    )

    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Index built successfully!")
    print(f"  Images: {len(manifest)}")
    print(f"  Embedding shape: {emb_matrix.shape}")
    print(f"  Output: {output_path}")
    print(f"\nFiles generated:")
    print(f"  {output_path / 'embeddings.npz'}")
    print(f"  {output_path / 'manifest.json'}")
    print(f"  {output_path / 'thumbnails/'} ({len(manifest)} thumbnails)")


if __name__ == "__main__":
    # Detect if running directly inside a Jupyter/Colab cell
    if 'ipykernel' in sys.modules:
        print("💡 Detected Jupyter/Colab notebook environment.")
        print("Using default paths. Modify these variables in the script if needed:")
        
        # ⚠️ CHANGE THESE IF NEEDED WHEN RUNNING IN COLAB ⚠️
        DATASET_DIR = "/content/tomato_dataset" # Path to your dataset folder
        ONNX_PATH = "/content/tomato_disease_efficientnet.onnx" # Path to your ONNX model
        OUTPUT_DIR = "similarity"
        
        print(f"  Dataset: {DATASET_DIR}")
        print(f"  ONNX: {ONNX_PATH}")
        print(f"  Output: {OUTPUT_DIR}\n")
        
        build_index(DATASET_DIR, ONNX_PATH, OUTPUT_DIR)
    else:
        parser = argparse.ArgumentParser(description="Build similarity index for tomato disease classifier")
        parser.add_argument("--dataset", required=True, help="Path to dataset root (contains class subdirectories)")
        parser.add_argument("--onnx", default="train-new/efficient-net/tomato_disease_efficientnet.onnx",
                            help="Path to EfficientNet ONNX model")
        parser.add_argument("--output", default="similarity", help="Output directory")
        args = parser.parse_args()

        build_index(args.dataset, args.onnx, args.output)
