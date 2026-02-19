"""
Bioplausible Deployment Utilities

Model export, serialization, and deployment utilities for production use.

Features:
- ONNX export for cross-platform deployment
- TorchScript compilation for optimized inference
- Model serialization/deserialization
- Inference optimization
- Batch prediction utilities
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class ModelInfo:
    """Metadata about an exported model."""
    model_name: str
    model_params: Dict[str, Any]
    optimizer_name: Optional[str]
    optimizer_params: Optional[Dict[str, Any]]
    training_metrics: Dict[str, float]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_parameters: int
    export_format: str
    export_path: str


class ModelExporter:
    """
    Export Bioplausible models for deployment.
    
    Supported formats:
    - ONNX: Cross-platform inference
    - TorchScript: PyTorch optimized inference
    - State dict: PyTorch checkpoint
    - JSON config: Model configuration
    
    Example usage:
        exporter = ModelExporter()
        
        # Export to multiple formats
        exporter.export(
            model=model,
            model_name='looped_mlp',
            model_params={'input_dim': 784, 'hidden_dim': 256, 'output_dim': 10},
            output_dir='./exports',
            formats=['onnx', 'torchscript', 'config'],
        )
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def export(
        self,
        model: nn.Module,
        model_name: str,
        model_params: Dict[str, Any],
        output_dir: str = './exports',
        formats: List[str] = None,
        optimizer: Optional[Any] = None,
        optimizer_name: Optional[str] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        input_shape: Tuple[int, ...] = (1, 784),
        verbose: bool = True,
    ) -> ModelInfo:
        """
        Export model to multiple formats.
        
        Args:
            model: Model to export.
            model_name: Name of the model.
            model_params: Model parameters.
            output_dir: Output directory for exports.
            formats: List of formats ['onnx', 'torchscript', 'config', 'state'].
            optimizer: Optional optimizer for state export.
            optimizer_name: Name of optimizer.
            optimizer_params: Optimizer parameters.
            training_metrics: Training metrics to save.
            input_shape: Example input shape for tracing.
            verbose: Print progress.
        
        Returns:
            ModelInfo with export details.
        """
        if formats is None:
            formats = ['onnx', 'torchscript', 'config', 'state']
        
        os.makedirs(output_dir, exist_ok=True)
        model = model.to(self.device)
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Export to each format
        export_paths = {}
        
        if 'onnx' in formats:
            try:
                path = self._export_onnx(model, output_dir, input_shape, verbose)
                export_paths['onnx'] = path
            except Exception as e:
                if verbose:
                    print(f"  ONNX export failed: {e}")
        
        if 'torchscript' in formats:
            try:
                path = self._export_torchscript(model, output_dir, input_shape, verbose)
                export_paths['torchscript'] = path
            except Exception as e:
                if verbose:
                    print(f"  TorchScript export failed: {e}")
        
        if 'config' in formats:
            path = self._export_config(
                model_name, model_params, optimizer_name, optimizer_params,
                training_metrics, input_shape, output_dir, verbose
            )
            export_paths['config'] = path
        
        if 'state' in formats:
            path = self._export_state(model, optimizer, output_dir, verbose)
            export_paths['state'] = path
        
        # Create model info
        info = ModelInfo(
            model_name=model_name,
            model_params=model_params,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            training_metrics=training_metrics or {},
            input_shape=input_shape,
            output_shape=self._get_output_shape(model, input_shape),
            num_parameters=num_params,
            export_format=', '.join(export_paths.keys()),
            export_path=output_dir,
        )
        
        if verbose:
            print(f"Exported {model_name} to {output_dir}")
            print(f"  Formats: {info.export_format}")
            print(f"  Parameters: {num_params:,}")
        
        return info
    
    def _export_onnx(
        self,
        model: nn.Module,
        output_dir: str,
        input_shape: Tuple[int, ...],
        verbose: bool,
    ) -> str:
        """Export to ONNX format."""
        path = os.path.join(output_dir, 'model.onnx')
        
        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)
        
        torch.onnx.export(
            model,
            dummy_input,
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
        )
        
        if verbose:
            print(f"  ✓ ONNX: {path}")
        
        return path
    
    def _export_torchscript(
        self,
        model: nn.Module,
        output_dir: str,
        input_shape: Tuple[int, ...],
        verbose: bool,
    ) -> str:
        """Export to TorchScript format."""
        path = os.path.join(output_dir, 'model.pt')
        
        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Trace the model
        traced = torch.jit.trace(model, dummy_input)
        traced.save(path)
        
        if verbose:
            print(f"  ✓ TorchScript: {path}")
        
        return path
    
    def _export_config(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        optimizer_name: Optional[str],
        optimizer_params: Optional[Dict[str, Any]],
        training_metrics: Optional[Dict[str, float]],
        input_shape: Tuple[int, ...],
        output_dir: str,
        verbose: bool,
    ) -> str:
        """Export model configuration to JSON."""
        path = os.path.join(output_dir, 'config.json')
        
        config = {
            'model_name': model_name,
            'model_params': model_params,
            'optimizer_name': optimizer_name,
            'optimizer_params': optimizer_params,
            'training_metrics': training_metrics,
            'input_shape': input_shape,
            'export_version': '1.0',
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        if verbose:
            print(f"  ✓ Config: {path}")
        
        return path
    
    def _export_state(
        self,
        model: nn.Module,
        optimizer: Optional[Any],
        output_dir: str,
        verbose: bool,
    ) -> str:
        """Export model and optimizer state."""
        path = os.path.join(output_dir, 'checkpoint.pt')
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        
        if verbose:
            print(f"  ✓ State: {path}")
        
        return path
    
    def _get_output_shape(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Get model output shape."""
        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        return tuple(output.shape)


class ModelLoader:
    """
    Load exported Bioplausible models.
    
    Example usage:
        loader = ModelLoader()
        
        # Load from config
        model, config = loader.load_from_config('./exports/config.json')
        
        # Load from checkpoint
        model = loader.load_from_checkpoint('./exports/checkpoint.pt', model_class)
        
        # Load ONNX for inference
        session = loader.load_onnx('./exports/model.onnx')
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def load_from_config(
        self,
        config_path: str,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load model from config file.
        
        Args:
            config_path: Path to config.json.
        
        Returns:
            Tuple of (model, config dict).
        """
        from bioplausible.zoo import ModelZoo
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_name = config['model_name']
        model_params = config['model_params']
        
        model = ModelZoo.get(model_name, **model_params)
        model = model.to(self.device)
        
        # Load state dict if available
        state_path = config_path.replace('config.json', 'checkpoint.pt')
        if os.path.exists(state_path):
            checkpoint = torch.load(state_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, config
    
    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        model_class: type,
        model_params: Dict[str, Any],
    ) -> nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint.pt.
            model_class: Model class.
            model_params: Model parameters.
        
        Returns:
            Loaded model.
        """
        model = model_class(**model_params)
        model = model.to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def load_onnx(
        self,
        onnx_path: str,
    ) -> Any:
        """
        Load ONNX model for inference.
        
        Args:
            onnx_path: Path to model.onnx.
        
        Returns:
            ONNX runtime session.
        """
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            return session
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")


class InferenceEngine:
    """
    Optimized inference engine for deployed models.
    
    Supports:
    - Batch prediction
    - Streaming prediction
    - Confidence scoring
    - Multiple input formats
    
    Example usage:
        engine = InferenceEngine.from_export('./exports')
        
        # Single prediction
        result = engine.predict(image)
        
        # Batch prediction
        results = engine.predict_batch(images)
        
        # With confidence
        pred, confidence = engine.predict_with_confidence(image)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'auto',
    ):
        self.model = model
        self.config = config
        self.device = device
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.input_shape = tuple(config.get('input_shape', (1, 784)))
    
    @classmethod
    def from_export(cls, export_dir: str, device: str = 'auto') -> 'InferenceEngine':
        """
        Create inference engine from export directory.
        
        Args:
            export_dir: Directory with config.json and checkpoint.pt.
            device: Device for inference.
        
        Returns:
            InferenceEngine instance.
        """
        loader = ModelLoader(device='cpu')
        config_path = os.path.join(export_dir, 'config.json')
        
        model, config = loader.load_from_config(config_path)
        
        return cls(model, config, device)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single prediction.
        
        Args:
            x: Input tensor.
        
        Returns:
            Prediction tensor.
        """
        x = x.to(self.device)
        
        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1:] != self.input_shape[1:]:
            x = x.view(-1, *self.input_shape[1:])
        
        output = self.model(x)
        return output
    
    @torch.no_grad()
    def predict_batch(
        self,
        xs: List[torch.Tensor],
        batch_size: int = 32,
    ) -> List[torch.Tensor]:
        """
        Batch prediction.
        
        Args:
            xs: List of input tensors.
            batch_size: Batch size for processing.
        
        Returns:
            List of predictions.
        """
        self.model.eval()
        predictions = []
        
        for i in range(0, len(xs), batch_size):
            batch = xs[i:i + batch_size]
            batched = torch.stack(batch).to(self.device)
            output = self.model(batched)
            predictions.extend(output.cpu().chunk(output.shape[0]))
        
        return predictions
    
    @torch.no_grad()
    def predict_with_confidence(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with confidence score.
        
        Args:
            x: Input tensor.
        
        Returns:
            Tuple of (prediction, confidence).
        """
        output = self.predict(x)
        
        # Get confidence from softmax
        probs = torch.softmax(output, dim=-1)
        confidence, pred = probs.max(dim=-1)
        
        return pred, confidence
    
    @torch.no_grad()
    def predict_class(
        self,
        x: torch.Tensor,
    ) -> int:
        """
        Get predicted class index.
        
        Args:
            x: Input tensor.
        
        Returns:
            Class index.
        """
        output = self.predict(x)
        return output.argmax(dim=-1).item()
    
    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor.
        
        Returns:
            Probability distribution.
        """
        output = self.predict(x)
        return torch.softmax(output, dim=-1)


def export_model(
    model: nn.Module,
    model_name: str,
    model_params: Dict[str, Any],
    output_dir: str = './exports',
    formats: List[str] = None,
    optimizer: Optional[Any] = None,
    training_metrics: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> ModelInfo:
    """
    Convenience function to export a model.
    
    Args:
        model: Model to export.
        model_name: Name of the model.
        model_params: Model parameters.
        output_dir: Output directory.
        formats: Export formats.
        optimizer: Optional optimizer.
        training_metrics: Training metrics.
        verbose: Print progress.
    
    Returns:
        ModelInfo with export details.
    """
    exporter = ModelExporter()
    return exporter.export(
        model=model,
        model_name=model_name,
        model_params=model_params,
        output_dir=output_dir,
        formats=formats,
        optimizer=optimizer,
        optimizer_params=None,
        training_metrics=training_metrics,
        verbose=verbose,
    )


def load_model(
    export_dir: str,
    device: str = 'auto',
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to load a model.
    
    Args:
        export_dir: Export directory with config.json.
        device: Device for model.
    
    Returns:
        Tuple of (model, config).
    """
    loader = ModelLoader(device='cpu')
    config_path = os.path.join(export_dir, 'config.json')
    return loader.load_from_config(config_path)


__all__ = [
    'ModelInfo',
    'ModelExporter',
    'ModelLoader',
    'InferenceEngine',
    'export_model',
    'load_model',
]
