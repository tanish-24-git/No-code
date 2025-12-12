"""
Model registry for tracking trained models and their metadata.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
from app.storage.object_store import object_store
from app.utils.logging import get_logger
from app.utils.exceptions import StorageException

logger = get_logger(__name__)


class ModelRegistry:
    """
    Model artifact management and versioning.
    Tracks model lineage, metadata, and export formats.
    """
    
    def __init__(self):
        self.store = object_store
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        base_model: str,
        dataset_id: str,
        training_config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a trained model with metadata.
        
        Args:
            run_id: Training run ID
            model_name: Model name/identifier
            base_model: Base model used for fine-tuning
            dataset_id: Dataset used for training
            training_config: Training configuration
            metrics: Evaluation metrics
            model_path: Path to model in object storage
        
        Returns:
            Model metadata dictionary
        """
        metadata = {
            "run_id": run_id,
            "model_name": model_name,
            "base_model": base_model,
            "dataset_id": dataset_id,
            "training_config": training_config,
            "metrics": metrics or {},
            "model_path": model_path,
            "created_at": datetime.utcnow().isoformat(),
            "version": self._generate_version(model_name)
        }
        
        # Save metadata to object storage
        metadata_path = f"{run_id}/model_metadata.json"
        self._save_metadata(metadata_path, metadata)
        
        logger.info(
            "Model registered",
            run_id=run_id,
            model_name=model_name,
            version=metadata["version"]
        )
        
        return metadata
    
    def add_export(
        self,
        run_id: str,
        export_format: str,
        export_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an export format to a model.
        
        Args:
            run_id: Training run ID
            export_format: Export format ('adapter', 'merged', 'gguf')
            export_path: Path to exported model
            metadata: Additional export metadata
        """
        # Load existing model metadata
        metadata_path = f"{run_id}/model_metadata.json"
        model_metadata = self._load_metadata(metadata_path)
        
        if not model_metadata:
            raise StorageException(f"Model metadata not found for run {run_id}")
        
        # Add export info
        if "exports" not in model_metadata:
            model_metadata["exports"] = {}
        
        model_metadata["exports"][export_format] = {
            "path": export_path,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save updated metadata
        self._save_metadata(metadata_path, model_metadata)
        
        logger.info(
            "Export added to model",
            run_id=run_id,
            export_format=export_format
        )
    
    def get_model_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by run ID."""
        metadata_path = f"{run_id}/model_metadata.json"
        return self._load_metadata(metadata_path)
    
    def list_models(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List all registered models."""
        # List all metadata files
        objects = self.store.list_objects(prefix=prefix, bucket_type='models')
        
        models = []
        for obj in objects:
            if obj.endswith('model_metadata.json'):
                metadata = self._load_metadata(obj)
                if metadata:
                    models.append(metadata)
        
        return models
    
    def generate_model_card(
        self,
        run_id: str,
        model_metadata: Dict[str, Any]
    ) -> str:
        """
        Generate a model card in Markdown format.
        
        Args:
            run_id: Training run ID
            model_metadata: Model metadata
        
        Returns:
            Model card content (Markdown)
        """
        card = f"""# {model_metadata['model_name']}

## Model Description

This model was fine-tuned from `{model_metadata['base_model']}` using the LLM Fine-Tuning Platform.

**Run ID:** `{run_id}`  
**Version:** `{model_metadata.get('version', 'N/A')}`  
**Created:** `{model_metadata.get('created_at', 'N/A')}`

## Training Configuration

```json
{json.dumps(model_metadata.get('training_config', {}), indent=2)}
```

## Evaluation Metrics

```json
{json.dumps(model_metadata.get('metrics', {}), indent=2)}
```

## Dataset

**Dataset ID:** `{model_metadata.get('dataset_id', 'N/A')}`

## Available Exports

"""
        
        exports = model_metadata.get('exports', {})
        if exports:
            for format_name, export_info in exports.items():
                card += f"- **{format_name}**: `{export_info['path']}`\n"
        else:
            card += "No exports available yet.\n"
        
        card += """
## Usage

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter (if using adapter export)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "path/to/adapter")
```

## Limitations and Biases

This model inherits any limitations and biases from the base model. Please review the base model's documentation for details.

## License

Inherits license from base model: `{base_model}`
""".format(base_model=model_metadata['base_model'])
        
        return card
    
    def _generate_version(self, model_name: str) -> str:
        """Generate version string for model."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"v{timestamp}"
    
    def _save_metadata(self, object_name: str, metadata: Dict[str, Any]):
        """Save metadata as JSON to object storage."""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(metadata, f, indent=2)
                temp_path = f.name
            
            self.store.upload_file(
                temp_path,
                object_name,
                bucket_type='models',
                content_type='application/json'
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
        
        except Exception as e:
            logger.error("Failed to save metadata", object_name=object_name, error=str(e))
            raise StorageException(f"Failed to save metadata: {str(e)}")
    
    def _load_metadata(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Load metadata from object storage."""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_path = f.name
            
            self.store.download_file(
                object_name,
                temp_path,
                bucket_type='models'
            )
            
            with open(temp_path, 'r') as f:
                metadata = json.load(f)
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            return metadata
        
        except Exception as e:
            logger.debug("Failed to load metadata", object_name=object_name, error=str(e))
            return None


# Global model registry instance
model_registry = ModelRegistry()
