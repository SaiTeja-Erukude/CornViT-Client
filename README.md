# CornViT Client Application

A modern web application for hierarchical corn kernel analysis using the CornViT Multi-Stage Convolutional Vision Transformer Framework.

## Features

- **Stage 1**: Purity Classification (Pure/Impure)
- **Stage 2**: Shape Classification (Flat/Round) - Only for Pure corn
- **Stage 3**: Embryo Orientation (Embryo Up/Embryo Down) - Only for Flat corn
- **Real-time Inference**: Upload images via drag-and-drop or file selection
- **Hierarchical Pipeline**: Conditional execution based on previous stage results
- **Confidence Scores**: View prediction probabilities for each class
- **Modern UI**: Responsive design with intuitive interface

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, falls back to CPU)
- 16GB RAM recommended for inference

## Installation

1. Clone the repository with the CvT submodule:
```bash
git clone <repository-url>
cd CornViT-Client
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the following directory structure:
```
CornViT-Client/
├── app.py                      # Flask application
├── inference_cvt13.py          # Model evaluation script
├── requirements.txt            # Python dependencies
├── models/
│   ├── stage1_model.pth        # Purity classification model
│   ├── stage2_model.pth        # Shape classification model
│   └── stage3_model.pth        # Embryo orientation model
├── CvT/                        # Microsoft CvT repository
│   ├── lib/
│   │   ├── models/
│   │   │   └── cls_cvt.py
│   │   └── config/
│   └── experiments/
│       └── imagenet/cvt/
│           └── cvt-13-384x384.yaml
├── templates/
│   └── index.html              # Frontend interface
└── static/
    ├── css/
    │   └── style.css           # Styles
    └── uploads/                # Temporary upload folder
```

## Usage

### Web Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload a corn kernel image using drag-and-drop or file selector

4. Click "Analyze Corn Image" to run the hierarchical inference

5. View detailed results including:
   - Predicted class for each stage
   - Confidence scores
   - Probability distributions
   - Skipped stages with reasons

### Model Evaluation

To evaluate models on a test dataset:

1. Configure paths in `inference_cvt13.py`:
```python
BASE_DIR = "path_to_CornViT"
MODEL_PATH = "metrics/RUN_NAME/train/best_model.pth"
TEST_DATA_DIR = "stage1/data/test"
```

2. Run evaluation:
```bash
python inference_cvt13.py
```

3. View generated metrics:
   - Confusion matrices
   - ROC curves
   - Precision-Recall curves
   - Per-class performance metrics
   - Confidence distributions
   - Detailed prediction CSV files

## Technical Details

### Model Architecture
- **Base Model**: CvT-13 (Convolutional Vision Transformer)
- **Input Size**: 384×384 RGB images
- **Preprocessing**: ImageNet normalization
- **Output**: Binary classification per stage (2 classes each)

### Framework
- **Backend**: Flask 3.1.2
- **Deep Learning**: PyTorch 2.9.1, torchvision 0.24.1
- **Vision Transformer**: Microsoft CvT with timm 1.0.22
- **Data Processing**: PIL, NumPy

### Inference Pipeline
- Automatic torch._six compatibility fixes
- Single-image prediction mode
- Softmax probabilities with confidence scores
- GPU acceleration when available

## Hierarchical Pipeline Logic

```
Stage 1: Purity Classification
    ├── Pure (Confidence: X%) → Stage 2: Shape Classification
    │                               ├── Flat (Confidence: Y%) → Stage 3: Embryo Orientation
    │                               │                               ├── Embryo Up (Confidence: Z%)
    │                               │                               └── Embryo Down (Confidence: Z%)
    │                               └── Round (Confidence: Y%) → END
    └── Impure (Confidence: X%) → Stages 2 & 3 Skipped
```

## API Endpoints

### `GET /`
Returns the main HTML interface

### `POST /predict`
Accepts multipart form data with image file
- **Input**: `file` (image/jpeg, image/png, max 16MB)
- **Output**: JSON with hierarchical predictions
```json
{
  "success": true,
  "results": {
    "stage1": {
      "name": "Purity Classification",
      "executed": true,
      "result": {
        "prediction": "Pure",
        "confidence": 0.9845,
        "probabilities": {"Pure": 0.9845, "Impure": 0.0155}
      }
    },
    "stage2": { ... },
    "stage3": { ... }
  }
}
```

### `GET /health`
Health check endpoint
- **Output**: Server status and model information

## Troubleshooting

### Common Issues

**"torch._six not found" error**:
- The app automatically patches this compatibility issue on startup

**CUDA out of memory**:
- Reduce batch size or use CPU mode
- Close other GPU-intensive applications

**Model loading fails**:
- Ensure all `.pth` files are in `models/` directory
- Check that CvT repository is properly configured

**Low confidence predictions**:
- Review `low_confidence_predictions.csv` from evaluation
- Consider retraining with more data


## Citation

Bibtex coming soon...
```