"""
Scikit-Learn Compatible Wrapper for EqProp Models

Allows using EqProp models in Scikit-Learn pipelines with .fit() and .predict().
Supports incremental learning via .partial_fit().
"""

import numpy as np
import torch

# Resilience against broken sklearn/pyarrow
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.multiclass import unique_labels
    from sklearn.utils.validation import (check_array, check_is_fitted,
                                          check_X_y)

    SKLEARN_AVAILABLE = True
except ImportError:
    # Dummy classes to prevent import crash
    class BaseEstimator:
        pass

    class ClassifierMixin:
        pass

    # Dummy utility functions
    def unique_labels(*args):
        return []

    def check_array(X, **kwargs):
        return X

    def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
        pass

    def check_X_y(X, y, **kwargs):
        return X, y

    SKLEARN_AVAILABLE = False
from torch.utils.data import DataLoader, TensorDataset

from .core import EqPropTrainer
from .models.factory import create_model
from .models.registry import MODEL_REGISTRY, get_model_spec


class EqPropClassifier(BaseEstimator, ClassifierMixin):
    """
    Equilibrium Propagation Classifier compatible with Scikit-Learn.

    Supports incremental learning via partial_fit().

    Parameters
    ----------
    model_name : str, default="EqProp MLP"
        Name of the model to use (see MODEL_REGISTRY).
    hidden_dim : int, default=256
        Number of neurons in the hidden layer.
    steps : int, default=30
        Number of equilibrium steps during training.
    learning_rate : float, default=0.001
        Learning rate for the optimizer.
    batch_size : int, default=128
        Batch size for training.
    epochs : int, default=10
        Number of training epochs (for fit()).
    use_spectral_norm : bool, default=True
        Whether to use spectral normalization (required for stability).
    device : str, default='cpu'
        Device to train on ('cpu' or 'cuda').
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to model factory.
    """

    def __init__(
        self,
        model_name="EqProp MLP",
        hidden_dim=256,
        steps=30,
        learning_rate=0.001,
        batch_size=128,
        epochs=10,
        use_spectral_norm=True,
        device=None,
        random_state=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_spectral_norm = use_spectral_norm
        self.device = device
        self.random_state = random_state
        self.kwargs = kwargs

        # Internal state
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.model_ = None
        self.trainer_ = None

    def _initialize(self, X, y=None, classes=None):
        """Initialize the model and trainer if not already initialized."""
        if self.model_ is not None:
            return

        # Determine classes
        if classes is not None:
            self.classes_ = unique_labels(classes)
        elif y is not None:
            self.classes_ = unique_labels(y)
        else:
            raise ValueError(
                "Classes must be provided for initialization if y is None."
            )

        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Determine device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create Model using Factory
        spec = get_model_spec(self.model_name)

        # Some logic to handle dimensionality.
        # create_model logic for vision usually assumes input_dim is flattened or channel logic handled
        # If X is (N, D), input_dim=D.

        # Check if kwargs override defaults
        factory_kwargs = self.kwargs.copy()

        self.model_ = create_model(
            spec=spec,
            input_dim=self.n_features_in_,
            output_dim=self.n_classes_,
            hidden_dim=self.hidden_dim,
            device=self.device,
            task_type="vision",  # Default to vision for tabular/image vectors
            **factory_kwargs,
        )

        # Handle manual overrides for things that create_model sets from spec
        # (Trainer will handle steps, but we can set max_steps on model too just in case)
        if hasattr(self.model_, "max_steps"):
            self.model_.max_steps = self.steps
        if hasattr(self.model_, "eq_steps"):
            self.model_.eq_steps = self.steps

        # Initialize Trainer
        self.trainer_ = EqPropTrainer(
            model=self.model_,
            task=None,
            task_type="vision",
            lr=self.learning_rate,
            device=self.device,
            use_compile=False,  # Disable compile for dynamic/sklearn usage
            steps=self.steps,
        )

    def fit(self, X, y):
        """
        Train the EqProp model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self._initialize(X, y)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train using fit method
        self.trainer_.fit(loader, epochs=self.epochs, progress_bar=False)

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incremental fit on a batch of samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        classes : array-like, optional
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)

        if self.model_ is None:
            self._initialize(X, y, classes=classes)

        # Verify classes match
        if self.classes_ is None and classes is None:
            raise ValueError("classes must be passed on the first call to partial_fit.")

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        self.trainer_.train_batch(X_tensor, y_tensor)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        X_tensor = torch.FloatTensor(X).to(self.trainer_.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)

        X_tensor = torch.FloatTensor(X).to(self.trainer_.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()
