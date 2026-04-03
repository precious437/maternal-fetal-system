"""
YOLOv8 Maternal Health Imaging Detection System
Trains a YOLOv8 model for ultrasound, maternal surgery, and fatality risk detection
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import yaml

# YOLOv8 and training libraries
from ultralytics import YOLO
import torch

class MaternalHealthYOLOTrainer:
    """Train YOLOv8 model for maternal health imaging"""
    
    def __init__(self, model_size="medium", project_name="maternal_health_detection"):
        self.model_size = model_size  # nano, small, medium, large, xlarge
        self.project_name = project_name
        self.base_dir = Path(project_name)
        self.datasets_dir = self.base_dir / "datasets"
        self.runs_dir = self.base_dir / "runs"
        self.models_dir = self.base_dir / "models"
        
        # Create directories
        for d in [self.datasets_dir, self.runs_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.gpu_available = torch.cuda.is_available()
        print(f"GPU Available: {self.gpu_available}")
        if self.gpu_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def create_dataset_structure(self):
        """Create YOLO-compatible dataset structure"""
        dataset = self.datasets_dir / "maternal_health"
        
        directories = [
            dataset / "images" / "train",
            dataset / "images" / "val",
            dataset / "images" / "test",
            dataset / "labels" / "train",
            dataset / "labels" / "val",
            dataset / "labels" / "test",
        ]
        
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)
        
        return dataset
    
    def create_dataset_yaml(self, dataset_path, class_names):
        """Create dataset.yaml for YOLO training"""
        yaml_content = {
            'path': str(dataset_path),
            'train': str(dataset_path / 'images' / 'train'),
            'val': str(dataset_path / 'images' / 'val'),
            'test': str(dataset_path / 'images' / 'test'),
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        return yaml_path
    
    def download_public_datasets(self):
        """
        Guide for downloading maternal health datasets from trusted sources
        Returns list of recommended dataset URLs
        """
        datasets = {
            'Ultrasound Imaging': {
                'HC18-MICCAI (Fetal Head Circumference)': 
                    'https://www.synapse.org/#!Synapse:syn3193805/wiki/217530',
                'MICCAI Fetal Echo Challenge': 
                    'https://www.synapse.org/#!Synapse:syn49712829/wiki/',
                'Fetal Imaging Datasets': 
                    'http://www.ipcai.info/challenge-2023/',
                'BACH Breast Cancer Dataset': 
                    'https://zenodo.org/record/4633574',
            },
            'Pregnancy & Maternal Health': {
                'Stanford Maternal Health': 
                    'https://stanfordmedicine.box.com/s/xxxx',
                'Kaggle Maternal Health Risk': 
                    'https://www.kaggle.com/datasets/ankur0899/maternal-health-risk-data',
                'WHOPrm Preeclampsia': 
                    'https://www.who.int/data/maternal-newborn-child',
            },
            'Surgical Imaging': {
                'MICCAI Surgical Workflow Challenge': 
                    'https://surgery.","readyai.org/',
                'CholecT45 Laparoscopic Surgery': 
                    'https://www.ai4da.org/site/assets/files/1346/cholecseg45_dataset.html',
                'OSIC Pulmonary Fibrosis': 
                    'https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression',
            }
        }
        
        return datasets
    
    def create_sample_labels(self, image_path, annotations):
        """
        Create YOLO label files (txt format)
        Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
        """
        label_path = image_path.with_suffix('.txt')
        
        with open(label_path, 'w') as f:
            for annotation in annotations:
                f.write(f"{annotation}\n")
    
    def prepare_synthetic_training_data(self):
        """Create synthetic training data for demonstration"""
        dataset = self.create_dataset_structure()
        
        print("Creating synthetic maternal health training data...")
        
        # Define classes
        classes = [
            'fetal_head',
            'fetal_spine', 
            'fetal_heart',
            'placenta',
            'amniotic_fluid',
            'surgical_instrument',
            'hemorrhage_indicator',
            'normal_anatomy',
            'abnormal_finding'
        ]
        
        # Create YAML
        yaml_path = self.create_dataset_yaml(dataset, classes)
        print(f"Dataset YAML created: {yaml_path}")
        
        # Create synthetic images
        print("Generating synthetic ultrasound images...")
        
        for split in ['train', 'val']:
            num_images = 80 if split == 'train' else 20
            
            for i in range(num_images):
                # Create synthetic ultrasound image (grayscale)
                img = np.random.randint(0, 255, (640, 480), dtype=np.uint8)
                
                # Add some ultrasound-like patterns
                y, x = np.ogrid[:640, :480]
                circle = (x - 240)**2 + (y - 320)**2 <= 80**2
                img[circle] = np.random.randint(100, 200, img[circle].shape)
                
                # Save image
                img_name = f"{split}_{i:03d}.jpg"
                img_path = dataset / 'images' / split / img_name
                cv2.imwrite(str(img_path), img)
                
                # Create labels (random annotations for demo)
                label_lines = []
                num_objects = np.random.randint(1, 4)
                
                for _ in range(num_objects):
                    class_id = np.random.randint(0, len(classes))
                    x_center = np.random.rand()
                    y_center = np.random.rand()
                    width = np.random.rand() * 0.3 + 0.1
                    height = np.random.rand() * 0.3 + 0.1
                    
                    label_lines.append(
                        f"{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}"
                    )
                
                # Save labels
                label_path = dataset / 'labels' / split / (img_name.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    for line in label_lines:
                        f.write(line + '\n')
        
        print(f"Synthetic data ready at: {dataset}")
        return dataset, classes, yaml_path
    
    def train_model(self, data_yaml, epochs=50, imgsz=640, batch_size=16):
        """Train YOLOv8 model"""
        
        print(f"\n{'='*60}")
        print(f"Training YOLOv8-{self.model_size.upper()}")
        print(f"{'='*60}")
        print(f"Model: yolov8{self.model_size[0]}.pt")  # n=nano, s=small, m=medium, l=large, x=xlarge
        print(f"Dataset: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch_size}")
        print(f"Device: {'GPU (CUDA)' if self.gpu_available else 'CPU'}")
        print(f"{'='*60}\n")
        
        # Load model
        model_name = f"yolov8{self.model_size[0]}.pt"  # n/s/m/l/x
        model = YOLO(model_name)
        
        # Train
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=0 if self.gpu_available else 'cpu',
            patience=20,
            project=str(self.runs_dir),
            name=f"maternal_health_{self.model_size}",
            save=True,
            save_period=5,
            val=True,
            plots=True,
            mosaic=1.0,
            augment=True,
            flipud=0.5,
            fliplr=0.5,
            degrees=10,
            translate=0.1,
            scale=0.5,
            verbose=True,
        )
        
        return model, results
    
    def evaluate_model(self, model, data_yaml):
        """Evaluate trained model"""
        print("\nEvaluating model...")
        metrics = model.val(data=str(data_yaml))
        
        print(f"\n{'='*60}")
        print("MODEL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def predict_on_image(self, model, image_path, conf_threshold=0.5):
        """Run inference on an image"""
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        return results
    
    def save_model(self, model, model_name="maternal_health_yolov8"):
        """Save trained model"""
        model_path = self.models_dir / f"{model_name}_{self.model_size}.pt"
        model.save(str(model_path))
        print(f"Model saved to: {model_path}")
        return model_path
    
    def export_model(self, model, format='onnx'):
        """Export model to different formats"""
        export_path = self.models_dir / f"maternal_health_{format}"
        model.export(format=format, imgsz=640)
        print(f"Model exported to {format} format")
        return export_path
    
    def get_training_summary(self):
        """Print training summary"""
        summary = f"""
        
{'='*70}
                    YOLOV8 MATERNAL HEALTH MODEL - SETUP COMPLETE
{'='*70}

MODEL CONFIGURATION:
  ├─ Model Size: {self.model_size.upper()}
  ├─ Model Type: YOLOv8
  ├─ GPU Support: {'✓ Yes (CUDA)' if self.gpu_available else '✗ CPU Only'}
  └─ Project Directory: {self.base_dir}

DATASET STRUCTURE:
  ├─ Images: {self.datasets_dir / 'maternal_health' / 'images'}
  ├─ Labels: {self.datasets_dir / 'maternal_health' / 'labels'}
  └─ Splits: train / val / test

DETECTION CLASSES (for training):
  ├─ fetal_head
  ├─ fetal_spine
  ├─ fetal_heart
  ├─ placenta
  ├─ amniotic_fluid
  ├─ surgical_instrument
  ├─ hemorrhage_indicator
  ├─ normal_anatomy
  └─ abnormal_finding

RECOMMENDED DATASETS (PUBLIC):
  1. MICCAI HC18 - Fetal Head Circumference
  2. Fetal Echocardiography Challenge
  3. Kaggle Maternal Health Risk
  4. CholecT45 - Surgical Videos
  5. Stanford Maternal Health Database

TRAINING PARAMETERS:
  ├─ Epochs: 50-100 (larger = better accuracy, slower)
  ├─ Batch Size: 16-32 (GPU dependent)
  ├─ Image Size: 640x640 (standard for YOLOv8)
  ├─ Augmentation: Enabled (flip, rotate, scale)
  └─ Device: {'GPU (Recommended)' if self.gpu_available else 'CPU (Slower)'}

NEXT STEPS:
  1. Download datasets from trusted websites
  2. Organize in: images/train, images/val, labels/train, labels/val
  3. Create data.yaml with class names
  4. Run trainer.train_model()
  5. Export model for production use

{'='*70}
        """
        return summary

def main():
    """Main training pipeline"""
    
    # Initialize trainer
    trainer = MaternalHealthYOLOTrainer(model_size="medium")
    
    print(trainer.get_training_summary())
    
    # Show available datasets
    print("\nRECOMMENDED PUBLIC DATASETS:\n")
    datasets = trainer.download_public_datasets()
    for category, sources in datasets.items():
        print(f"\n{category}:")
        for name, url in sources.items():
            print(f"  • {name}: {url}")
    
    # Prepare synthetic data for demo
    print("\n" + "="*70)
    dataset, classes, yaml_path = trainer.prepare_synthetic_training_data()
    
    # Train model
    print("\nTo train the model, run:")
    print(f"""
    model, results = trainer.train_model(
        data_yaml='{yaml_path}',
        epochs=50,
        imgsz=640,
        batch_size=16
    )
    """)
    
    print("\nTo evaluate, run:")
    print(f"    metrics = trainer.evaluate_model(model, '{yaml_path}')")
    
    print("\nTo use for inference:")
    print(f"    results = trainer.predict_on_image(model, 'image.jpg')")

if __name__ == "__main__":
    main()
