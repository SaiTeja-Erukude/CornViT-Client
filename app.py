import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
import warnings

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CVT_REPO_PATH = os.path.join(BASE_DIR, "CvT")
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fix torch._six compatibility in CvT library
cls_cvt_path = os.path.join(CVT_REPO_PATH, "lib", "models", "cls_cvt.py")
if os.path.exists(cls_cvt_path):
    with open(cls_cvt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "from torch._six import container_abcs" in content:
        content = content.replace(
            "from torch._six import container_abcs",
            "import collections.abc as container_abcs"
        )
        content = content.replace(
            "or pretrained_layers[0] is '*'",
            "or pretrained_layers[0] == '*'"
        )
        with open(cls_cvt_path, 'w', encoding='utf-8') as f:
            f.write(content)

# Add CvT to path
sys.path.insert(0, CVT_REPO_PATH)

warnings.filterwarnings('ignore', category=SyntaxWarning)
from lib.models import cls_cvt
from lib.config import config, update_config

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Stage configurations
STAGE_CONFIGS = {
    'stage1': {
        'model_path': os.path.join(MODEL_DIR, 'stage1_model.pth'),
        'classes': ['Pure', 'Impure'],
        'name': 'Purity Classification'
    },
    'stage2': {
        'model_path': os.path.join(MODEL_DIR, 'stage2_model.pth'),
        'classes': ['Flat', 'Round'],
        'name': 'Shape Classification'
    },
    'stage3': {
        'model_path': os.path.join(MODEL_DIR, 'stage3_model.pth'),
        'classes': ['Embryo Up', 'Embryo Down'],
        'name': 'Embryo Orientation'
    }
}

# Global models dictionary
models = {}


def load_model(model_path):
    """Load a CvT-13 model"""
    config_path = os.path.join(CVT_REPO_PATH, "experiments", "imagenet", "cvt", "cvt-13-384x384.yaml")
    
    config.defrost()
    config.merge_from_file(config_path)
    config.MODEL.NUM_CLASSES = 2  # Binary classification
    config.MODEL.PRETRAINED = ''
    config.freeze()
    
    model = cls_cvt.get_cls_model(config)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    return model


def load_all_models():
    """Load all three stage models"""
    print("Loading models...")
    for stage, config in STAGE_CONFIGS.items():
        print(f"  Loading {stage}: {config['name']}")
        models[stage] = load_model(config['model_path'])
    print("✅ All models loaded successfully!")


def preprocess_image(image_path):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image_tensor


def predict_single_stage(model, image_tensor, class_names):
    """Run prediction on a single stage"""
    with torch.no_grad():
        output = model(image_tensor)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        pred_idx = predicted.item()
        pred_class = class_names[pred_idx]
        pred_confidence = confidence.item()
        
        probs = {class_names[i]: float(probabilities[0][i].item()) 
                for i in range(len(class_names))}
        
        return {
            'prediction': pred_class,
            'confidence': pred_confidence,
            'probabilities': probs
        }


def hierarchical_inference(image_path):
    """
    Hierarchical inference pipeline:
    Stage 1 (Purity) -> If Pure -> Stage 2 (Shape) -> If Flat -> Stage 3 (Embryo)
    """
    image_tensor = preprocess_image(image_path)
    results = {}
    
    # Stage 1: Purity Classification
    stage1_result = predict_single_stage(
        models['stage1'], 
        image_tensor, 
        STAGE_CONFIGS['stage1']['classes']
    )
    results['stage1'] = {
        'name': STAGE_CONFIGS['stage1']['name'],
        'result': stage1_result,
        'executed': True
    }
    
    # Stage 2: Shape Classification (only if Pure)
    if stage1_result['prediction'] == 'Pure':
        stage2_result = predict_single_stage(
            models['stage2'], 
            image_tensor, 
            STAGE_CONFIGS['stage2']['classes']
        )
        results['stage2'] = {
            'name': STAGE_CONFIGS['stage2']['name'],
            'result': stage2_result,
            'executed': True
        }
        
        # Stage 3: Embryo Orientation (only if Flat)
        if stage2_result['prediction'] == 'Flat':
            stage3_result = predict_single_stage(
                models['stage3'], 
                image_tensor, 
                STAGE_CONFIGS['stage3']['classes']
            )
            results['stage3'] = {
                'name': STAGE_CONFIGS['stage3']['name'],
                'result': stage3_result,
                'executed': True
            }
        else:
            results['stage3'] = {
                'name': STAGE_CONFIGS['stage3']['name'],
                'result': None,
                'executed': False,
                'reason': 'Skipped (corn is Round, not Flat)'
            }
    else:
        results['stage2'] = {
            'name': STAGE_CONFIGS['stage2']['name'],
            'result': None,
            'executed': False,
            'reason': 'Skipped (corn is Impure)'
        }
        results['stage3'] = {
            'name': STAGE_CONFIGS['stage3']['name'],
            'result': None,
            'executed': False,
            'reason': 'Skipped (corn is Impure)'
        }
    
    return results


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run hierarchical inference
            results = hierarchical_inference(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'results': results
            })
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) == 3,
        'device': DEVICE
    })


if __name__ == '__main__':
    # Load all models at startup
    load_all_models()
    
    print("\n" + "="*60)
    print("🌽 CornViT Multi-Stage Inference Server")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print("\nHierarchical Pipeline:")
    print("  Stage 1: Purity (Pure/Impure)")
    print("  Stage 2: Shape (Flat/Round) - if Pure")
    print("  Stage 3: Embryo Orientation (Up/Down) - if Flat")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

