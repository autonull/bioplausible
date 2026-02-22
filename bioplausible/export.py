"""
Export utilities for Bioplausible models.
"""

from typing import List, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def export_to_onnx(model, input_sample, path):
    """Export model to ONNX format."""
    model.eval()

    # Trace/Script if needed, but standard export usually works for simple models.
    # For EqProp, we export the inference pass (forward).

    # Need to handle custom args like steps=... if they are in forward signature but not used in trace?
    # Torch ONNX export handles standard forward.

    torch.onnx.export(
        model,
        input_sample,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def export_to_torchscript(model, input_sample, path):
    """Export model to TorchScript (JIT).
    
    Note: torch.jit is deprecated in Python 3.14+.
    Consider using torch.compile for new projects.
    """
    import warnings
    warnings.warn(
        "torch.jit is deprecated in Python 3.14+. "
        "Consider using torch.compile for new projects.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    model.eval()
    traced_script_module = torch.jit.trace(model, input_sample)
    traced_script_module.save(path)


# --- Serving Logic ---

app = FastAPI(title="Bioplausible Inference API")
# Global model instance for FastAPI routes.
# Note: Using global state is necessary here for the simple FastAPI integration.
model_instance = None


class InferenceRequest(BaseModel):
    data: List[float]
    shape: Optional[List[int]] = None


@app.post("/predict")
def predict(request: InferenceRequest):
    if not model_instance:
        return {"error": "No model loaded"}

    try:
        data = np.array(request.data, dtype=np.float32)
        if request.shape:
            data = data.reshape(request.shape)
        else:
            # Try to infer shape or assume flat
            if hasattr(model_instance, "input_dim"):
                if len(data.shape) == 1 and data.shape[0] == model_instance.input_dim:
                    data = data.reshape(1, -1)
            elif "Conv" in type(model_instance).__name__:
                # Assume MNIST/CIFAR single image flat or shaped
                pass

        # Convert to tensor
        tensor = torch.from_numpy(data)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # Add batch dim

        device = next(model_instance.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model_instance(tensor)

        return {"output": output.cpu().tolist()}

    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": str(type(model_instance).__name__) if model_instance else "None",
    }


def serve_model(model, host="0.0.0.0", port=8000):
    """Run a FastAPI server for the model."""
    global model_instance
    model_instance = model
    model_instance.eval()

    # Run server
    uvicorn.run(app, host=host, port=port, log_level="info")
