import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
import seaborn as sns
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR        = "path_to_CornViT"

# Path to the Microsoft CvT repository
CVT_REPO_PATH = f"{BASE_DIR}/CvT"

# Model configuration
IMG_SIZE = 384
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN = "cvt13_run_2025xxxx_xxxxxx"

# Path to trained model
MODEL_PATH = f"metrics/{RUN}/train/best_model.pth"

# Test data folder (should have subfolders for each class like train/val structure)
TEST_DATA_DIR = f"{BASE_DIR}/stage1/data/test"

# Class names (update these to match your dataset)
CLASS_NAMES = ["Pure", "Impure"]

# Output directory for evaluation results (within the same metrics folder)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EVAL_OUTPUT_DIR = f"metrics/{RUN}/evals/eval_{timestamp}"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETUP: Import CvT model
# ============================================================

# Fix torch._six compatibility
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

sys.path.insert(0, CVT_REPO_PATH)

import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

from lib.models import cls_cvt
from lib.config import config, update_config


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path, config_path=None):
    """Load the trained CvT model"""
    
    # Load config
    if config_path is None:
        config_path = os.path.join(CVT_REPO_PATH, "experiments", "imagenet", "cvt", "cvt-13-384x384.yaml")
    
    config.defrost()
    config.merge_from_file(config_path)
    config.MODEL.NUM_CLASSES = NUM_CLASSES
    config.MODEL.PRETRAINED = ''
    config.freeze()
    
    # Create model
    model = cls_cvt.get_cls_model(config)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    return model


# ============================================================
# DATA LOADING
# ============================================================

def get_test_dataloader(test_dir, batch_size=32):
    """Create test dataloader"""
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✅ Test dataset loaded: {len(test_dataset)} images")
    print(f"   Classes: {test_dataset.classes}")
    return test_loader, test_dataset


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, test_loader, test_dataset):
    """
    Evaluate model with single image predictions
    
    Returns:
        all_preds: Predicted class labels
        all_labels: Ground truth labels
        all_probs: Predicted probabilities for all classes
        all_confidences: Confidence scores
        image_paths: List of image paths
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []
    image_paths = []
    
    print("\n🔍 Running single-image inference on test set...")
    
    # Process each image individually
    total_images = len(test_dataset)
    
    for idx in range(total_images):
        # Get single image and label
        image, label = test_dataset[idx]
        img_path, _ = test_dataset.samples[idx]
        
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Forward pass
            output = model(image)
            
            # Ensure output has correct shape
            if output.dim() == 1:
                output = output.unsqueeze(0)
                
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Collect results
            all_preds.append(predicted.item())
            all_labels.append(label)
            all_probs.append(probabilities.cpu().numpy()[0])
            all_confidences.append(confidence.item())
            image_paths.append(img_path)
        
        # Progress update
        if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
            print(f"   Processed {idx + 1}/{total_images} images...")
    
    print(f"✅ Inference complete: {len(all_preds)} predictions")
    
    return (np.array(all_preds), np.array(all_labels), np.array(all_probs), 
            np.array(all_confidences), image_paths)


# ============================================================
# METRICS CALCULATION
# ============================================================

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate all classification metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(CLASS_NAMES):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i])
        }
    
    # ROC-AUC (for binary and multi-class)
    if NUM_CLASSES == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        metrics['average_precision'] = average_precision_score(y_true, y_probs[:, 1])
    else:
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_probs, multi_class='ovo', average='macro')
    
    return metrics


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if NUM_CLASSES == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
    else:
        # Multi-class (one-vs-rest)
        for i, class_name in enumerate(CLASS_NAMES):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 ROC curve saved to: {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_probs, save_path):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if NUM_CLASSES == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        avg_precision = average_precision_score(y_true, y_probs[:, 1])
        
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
    else:
        # Multi-class
        for i, class_name in enumerate(CLASS_NAMES):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, i])
            avg_precision = average_precision_score(y_true_binary, y_probs[:, i])
            ax.plot(recall, precision, lw=2, 
                   label=f'{class_name} (AP = {avg_precision:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Precision-Recall curve saved to: {save_path}")
    plt.close()


def plot_class_distribution(y_true, y_pred, save_path):
    """Plot class distribution comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # True distribution
    true_counts = [np.sum(y_true == i) for i in range(NUM_CLASSES)]
    axes[0].bar(CLASS_NAMES, true_counts, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for i, count in enumerate(true_counts):
        axes[0].text(i, count + max(true_counts)*0.01, str(count), 
                    ha='center', va='bottom', fontweight='bold')
    
    # Predicted distribution
    pred_counts = [np.sum(y_pred == i) for i in range(NUM_CLASSES)]
    axes[1].bar(CLASS_NAMES, pred_counts, color='coral', alpha=0.7)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for i, count in enumerate(pred_counts):
        axes[1].text(i, count + max(pred_counts)*0.01, str(count), 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Class distribution saved to: {save_path}")
    plt.close()


def plot_per_class_metrics(metrics, save_path):
    """Plot per-class performance metrics"""
    classes = list(metrics['per_class'].keys())
    precision_vals = [metrics['per_class'][c]['precision'] for c in classes]
    recall_vals = [metrics['per_class'][c]['recall'] for c in classes]
    f1_vals = [metrics['per_class'][c]['f1_score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, recall_vals, width, label='Recall', color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='lightgreen', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Per-class metrics saved to: {save_path}")
    plt.close()


def plot_confidence_distribution(y_true, y_pred, confidences, save_path):
    """Plot confidence score distribution for correct vs incorrect predictions"""
    # Confidence scores are already extracted
    correct = (y_true == y_pred)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    axes[0].hist(confidences[correct], bins=50, alpha=0.7, label='Correct', 
                color='green', edgecolor='black')
    axes[0].hist(confidences[~correct], bins=50, alpha=0.7, label='Incorrect', 
                color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect Predictions', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data_to_plot = [confidences[correct], confidences[~correct]]
    box = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], 
                          patch_artist=True, showmeans=True)
    box['boxes'][0].set_facecolor('lightgreen')
    box['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Confidence Score', fontsize=12)
    axes[1].set_title('Confidence Score Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add statistics
    correct_mean = np.mean(confidences[correct])
    incorrect_mean = np.mean(confidences[~correct]) if (~correct).sum() > 0 else 0
    axes[1].text(1, correct_mean, f'μ={correct_mean:.3f}', 
                ha='right', va='center', fontweight='bold', fontsize=10)
    if (~correct).sum() > 0:
        axes[1].text(2, incorrect_mean, f'μ={incorrect_mean:.3f}', 
                    ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confidence distribution saved to: {save_path}")
    plt.close()


# ============================================================
# RESULTS SAVING
# ============================================================

def save_predictions_to_csv(image_paths, y_true, y_pred, y_probs, confidences, save_path):
    """Save detailed predictions to CSV"""
    results = []
    
    for img_path, true_label, pred, probs, conf in zip(image_paths, y_true, y_pred, y_probs, confidences):
        result = {
            'image_path': img_path,
            'image_name': os.path.basename(img_path),
            'true_label': CLASS_NAMES[true_label],
            'true_label_idx': true_label,
            'predicted_label': CLASS_NAMES[pred],
            'predicted_label_idx': pred,
            'confidence': conf,
            'correct': pred == true_label
        }
        
        # Add probabilities for each class
        for i, class_name in enumerate(CLASS_NAMES):
            result[f'prob_{class_name}'] = probs[i]
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"💾 Predictions saved to: {save_path}")
    
    # Print some statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Total images: {len(df)}")
    print(f"   Correct predictions: {df['correct'].sum()} ({df['correct'].sum()/len(df)*100:.2f}%)")
    print(f"   Incorrect predictions: {(~df['correct']).sum()} ({(~df['correct']).sum()/len(df)*100:.2f}%)")
    print(f"   Average confidence: {df['confidence'].mean():.4f}")
    print(f"   Confidence on correct: {df[df['correct']]['confidence'].mean():.4f}")
    print(f"   Confidence on incorrect: {df[~df['correct']]['confidence'].mean():.4f}" if (~df['correct']).sum() > 0 else "")
    
    return df


def save_metrics_json(metrics, save_path):
    """Save metrics to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Metrics saved to: {save_path}")


def generate_classification_report_file(y_true, y_pred, save_path):
    """Generate and save sklearn classification report"""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"📄 Classification report saved to: {save_path}")


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*60)
    print("CvT-13 MODEL EVALUATION PIPELINE")
    print("Single Image Prediction Mode")
    print("="*60 + "\n")
    
    # Load model
    print("📦 Loading model...")
    model = load_model(MODEL_PATH)
    
    # Load test data
    print("\n📂 Loading test data...")
    test_loader, test_dataset = get_test_dataloader(TEST_DATA_DIR, batch_size=1)
    
    # Run evaluation with single image predictions
    print("\n🔍 Evaluating model (single image predictions)...")
    y_pred, y_true, y_probs, confidences, image_paths = evaluate_model(model, test_loader, test_dataset)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_probs)
    
    # Print key metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Images Evaluated: {len(y_pred)}")
    print(f"Accuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"Precision (Macro):  {metrics['precision_macro']*100:.2f}%")
    print(f"Recall (Macro):     {metrics['recall_macro']*100:.2f}%")
    print(f"F1-Score (Macro):   {metrics['f1_macro']*100:.2f}%")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print("\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']*100:.2f}%")
        print(f"    Recall:    {class_metrics['recall']*100:.2f}%")
        print(f"    F1-Score:  {class_metrics['f1_score']*100:.2f}%")
    print("="*60)
    
    # Generate all visualizations
    print("\n📊 Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, 
                         os.path.join(EVAL_OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_true, y_probs, 
                  os.path.join(EVAL_OUTPUT_DIR, "roc_curve.png"))
    plot_precision_recall_curve(y_true, y_probs, 
                               os.path.join(EVAL_OUTPUT_DIR, "precision_recall_curve.png"))
    plot_class_distribution(y_true, y_pred, 
                          os.path.join(EVAL_OUTPUT_DIR, "class_distribution.png"))
    plot_per_class_metrics(metrics, 
                          os.path.join(EVAL_OUTPUT_DIR, "per_class_metrics.png"))
    plot_confidence_distribution(y_true, y_pred, confidences,
                                os.path.join(EVAL_OUTPUT_DIR, "confidence_distribution.png"))
    
    # Save results
    print("\n💾 Saving results...")
    df = save_predictions_to_csv(image_paths, y_true, y_pred, y_probs, confidences,
                                os.path.join(EVAL_OUTPUT_DIR, "predictions.csv"))
    save_metrics_json(metrics, 
                     os.path.join(EVAL_OUTPUT_DIR, "metrics.json"))
    generate_classification_report_file(y_true, y_pred,
                                       os.path.join(EVAL_OUTPUT_DIR, "classification_report.txt"))
    
    # Save misclassified images list
    misclassified = df[~df['correct']]
    if len(misclassified) > 0:
        misclassified_path = os.path.join(EVAL_OUTPUT_DIR, "misclassified_images.csv")
        misclassified.to_csv(misclassified_path, index=False)
        print(f"⚠️  Misclassified images saved to: {misclassified_path}")
        print(f"   Total misclassified: {len(misclassified)}")
    
    # Save low confidence predictions
    low_conf_threshold = 0.7
    low_confidence = df[df['confidence'] < low_conf_threshold]
    if len(low_confidence) > 0:
        low_conf_path = os.path.join(EVAL_OUTPUT_DIR, "low_confidence_predictions.csv")
        low_confidence.to_csv(low_conf_path, index=False)
        print(f"⚠️  Low confidence predictions saved to: {low_conf_path}")
        print(f"   Total with confidence < {low_conf_threshold}: {len(low_confidence)}")
    
    print("\n" + "="*60)
    print(f"✅ Evaluation complete!")
    print(f"📁 All results saved to: {EVAL_OUTPUT_DIR}")
    print("="*60 + "\n")
    
    print("Generated files:")
    print("  📊 confusion_matrix.png - Confusion matrix visualization")
    print("  📊 roc_curve.png - ROC curve")
    print("  📊 precision_recall_curve.png - Precision-Recall curve")
    print("  📊 class_distribution.png - Class distribution comparison")
    print("  📊 per_class_metrics.png - Per-class performance")
    print("  📊 confidence_distribution.png - Confidence analysis")
    print("  💾 predictions.csv - Detailed predictions for each image")
    print("  💾 misclassified_images.csv - List of incorrectly classified images")
    print("  💾 low_confidence_predictions.csv - Predictions with low confidence")
    print("  💾 metrics.json - All metrics in JSON format")
    print("  📄 classification_report.txt - Sklearn classification report")


if __name__ == '__main__':
    main()