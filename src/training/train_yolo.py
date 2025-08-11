from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import os

class YOLOTrainer:
    def __init__(self, config_path='config.yaml'):
        """
        Initialize YOLO trainer
        """

        self.config_path = config_path
        self.load_config()
        self.setup_training()

    def load_config(self):
        """
        Load model configuration from YAML file
        """
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                print("CUSTOM CONFIG loaded successfully.")
        except Exception as e:
            print("Using DEFAULT CONFIG as no config file was found.")

            self.config = self.get_default_config()

    def get_default_config(self):
        """Default configuration for YOLO training."""
        return {
            'model': {
                'version': 'yolo11n',  # Usando YOLO11 más moderno
                'pretrained': True,
                'image_size': 640,
                'batch_size': 16,
                'epochs': 50,
                'patience': 20
            },
            'training': {
                'device': 'auto',
                'workers': 6,
                'learning_rate': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005
            },
            'paths': {
                'data_yaml': 'data/data.yaml',
                'results': 'results/training'
            }
        }
    
    def resolve_project_path(self, relative_path):
        """
        Converts relative project path to absolute path
        """
        path_obj = Path(relative_path)
        if not path_obj.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            return project_root / path_obj
        return path_obj
    
    def setup_training(self):
        """
        Prepare the YOLO model for training
        """
        results_dir = results_dir = self.resolve_project_path(self.config['paths']['results'])
        results_dir.mkdir(parents=True, exist_ok=True)

        self.device = self.check_device()
        print("Training on device:", self.device)

        self.model = self.load_model()

    def check_device(self):
        """
        Check if CUDA is available and return the appropriate device
        """
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA is available. Using GPU for training.")
        else:
            device = 'cpu'
            print("CUDA is not available. Using CPU for training.")
        
        return device
    
    def load_model(self):
        """
        Load YOLO model
        """
        model_version = self.config['model']['version']

        if self.config['model']['pretrained']:
            model = YOLO(f"{model_version}.pt")
            print(f"Loaded pretrained model: {model_version}")
        else:
            model = YOLO(f"{model_version}.yaml")
            print(f"Creating model from scratch: {model_version}")

        return model
    
    def create_data_yaml(self):
        """
        Create data.yaml file if it does not exist
        """
        data_yaml_path = self.resolve_project_path(self.config['paths']['data_yaml'])

        if not data_yaml_path.exists():
            
            data_config = {
                'path': 'data',
                'train': 'VisDrone2019-DET-train/images',
                'val': 'VisDrone2019-DET-val/images',
                'test': 'VisDrone2019-DET-test-dev/images',
                'nc': 10,
                'names': [
                    'pedestrian', 'people', 'bicycle', 'car', 'van',
                    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
                ]
            }
            
            data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            print(f"✓ Archivo data.yaml creado: {data_yaml_path}")
        else:
            print(f"✓ Archivo data.yaml encontrado: {data_yaml_path}")

        return data_yaml_path
    
    def train(self):
        """
        Train the YOLO model
        """
        data_yaml = self.create_data_yaml()

        train_params = {
            'data': str(data_yaml),
            'epochs': self.config['model']['epochs'],
            'batch': self.config['model']['batch_size'],
            'imgsz': self.config['model']['image_size'],
            'device': self.device,
            'workers': self.config['training']['workers'],
            'lr0': self.config['training']['learning_rate'],
            'momentum': self.config['training']['momentum'],
            'weight_decay': self.config['training']['weight_decay'],
            'patience': self.config['model']['patience'],
            'project': str(self.resolve_project_path(self.config['paths']['results'])),
            'name': 'drone_traffic_model',
            'save_period': 10,
            'plots': True,
            'verbose': True
        }

        print("Training params:")
        for key, value in train_params.items():
            print(f"  {key}: {value}")

        try:
            results = self.model.train(**train_params)
            print("Training completed successfully.")
            return results
        except Exception as e:
            print("Error during training:", e)
            return None
        
    def validate_model(self, model_path=None):
        """
        Validate the trained YOLO model on the validation dataset
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        print("Validating model...")
        
        data_yaml_path = self.resolve_project_path(self.config['paths']['data_yaml'])
        
        val_results = model.val(
            data=str(data_yaml_path),
            imgsz=self.config['model']['image_size']
        )
        
        return val_results

def main():
    """Función principal"""
    print("=" * 60)
    print("🚁 DRONE TRAFFIC MONITORING - YOLO TRAINING")
    print("=" * 60)
    
    trainer = YOLOTrainer()    

    results = trainer.train()
    
    if results:
        print("\n" + "=" * 60)
        print("📊 TRAINING SUMMARY")
        print("=" * 60)
        print(f"✅ Training completed successfully.")
        print(f"📁 Results saved in: {trainer.config['paths']['results']}")
        print(f"🎯 Best model: {results.save_dir}/weights/best.pt")
        
        print("\n🔍 Validating final model...")
        trainer.validate_model()
    
if __name__ == "__main__":
    main()
