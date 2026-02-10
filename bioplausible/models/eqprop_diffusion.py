import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.models.modern_conv_eqprop import SimpleConvEqProp
from bioplausible.models.registry import register_model


@register_model("eqprop_diffusion")
class EqPropDiffusion(nn.Module):
    """
    Equilibrium Propagation Diffusion Model.

    Hypothesis: Denoising diffusion is energy minimization.
    Energy Formulation: E(x,t) = ||x - Denoise(x_t,t)||² + λR(x)

    This model predicts the clean image x_0 from x_t.
    """

    def __init__(self, img_channels=1, hidden_channels=64, gradient_method="bptt"):
        super().__init__()
        # Input channels + 1 for time embedding (concatenated as a channel)
        # Using SimpleConvEqProp allows Triton acceleration
        # Use pool_output=False for spatial output (Dense prediction)
        self.denoiser = SimpleConvEqProp(
            input_channels=img_channels + 1,
            hidden_channels=hidden_channels,
            output_dim=img_channels,  # Output channels
            pool_output=False,
            use_spectral_norm=True,
            gradient_method=gradient_method,
        )
        self.img_channels = img_channels

        # Register noise schedule buffers
        T = 1000
        self.T = T
        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_variance", posterior_variance)

    @classmethod
    def build(
        cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs
    ):
        # input_dim is interpreted as channels for vision tasks
        channels = input_dim if input_dim is not None else 1

        # Heuristic for flattened inputs (e.g., from TrialRunner)
        if channels == 784:  # MNIST flattened
            channels = 1
        elif channels == 3072:  # CIFAR-10 flattened
            channels = 3
        elif channels > 10:
            # Generic heuristic: check if square (grayscale) or 3*square (RGB)
            side = int(channels**0.5)
            if side * side == channels:
                channels = 1
            elif (channels % 3 == 0) and (
                int((channels / 3) ** 0.5) ** 2 * 3 == channels
            ):
                channels = 3

        return cls(img_channels=channels, hidden_channels=hidden_dim).to(device)

    def train_step(self, x, y=None):
        """
        Training step for SupervisedTrainer.
        x: Clean images [B, C, H, W]
        y: Labels (ignored)
        """
        device = x.device
        batch_size = x.shape[0]

        # Sample time steps
        t = torch.randint(0, self.T, (batch_size,), device=device).long()

        # Add noise
        noise = torch.randn_like(x)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_omab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        x_noisy = sqrt_ab * x + sqrt_omab * noise

        # Predict clean image (x_0)
        pred = self(x_noisy, t)

        # Compute Loss (MSE against clean image)
        loss = F.mse_loss(pred, x)

        # Optimization handled by Trainer usually, but if manual:
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item()}

    def val_step(self, x, y=None):
        """
        Validation step for SupervisedTrainer.
        Computes validation loss (MSE) and a proxy accuracy score.
        """
        device = x.device
        batch_size = x.shape[0]

        # Sample time steps
        t = torch.randint(0, self.T, (batch_size,), device=device).long()

        # Add noise
        noise = torch.randn_like(x)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_omab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        x_noisy = sqrt_ab * x + sqrt_omab * noise

        # Predict clean image (x_0)
        # Ensure we are in eval mode if not already (though val_step implies it)
        # but eqprop might need gradients? No, val_step is usually no_grad.
        pred = self(x_noisy, t)

        # Compute Loss
        loss = F.mse_loss(pred, x).item()

        # Proxy Accuracy for AutoScientist (maximize this)
        # 1.0 / (1.0 + loss) maps 0 -> 1, inf -> 0
        accuracy = 1.0 / (1.0 + loss)

        return {"loss": loss, "accuracy": accuracy}

    def predict_x0(self, x_t, t):
        """Predict x_0 given x_t and t."""
        # Embed time: simple broadcast/concat
        batch_size, _, h, w = x_t.shape

        # Normalize t to [0,1] for embedding
        t_norm = t.float() / self.T
        t_emb = t_norm.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)

        x_input = torch.cat([x_t, t_emb], dim=1)
        return self.denoiser(x_input)

    def denoise_step(self, x_t, t_norm, steps=30):
        """
        Perform a single denoise step with explicit control over equilibrium settling steps.
        Used for iterative inference or analysis where we want to control precision.

        Args:
            x_t: Noisy input [B, C, H, W]
            t_norm: Normalized time [B] or [B, 1, 1, 1] (0.0 to 1.0)
            steps: Number of equilibrium steps for the denoiser
        """
        batch_size, _, h, w = x_t.shape

        # Ensure t_norm is broadcastable
        if t_norm.dim() == 1:
            t_emb = t_norm.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        else:
            t_emb = t_norm.expand(batch_size, 1, h, w)

        x_input = torch.cat([x_t, t_emb], dim=1)

        # Forward through EqProp denoiser with specified settling steps
        return self.denoiser(x_input, steps=steps)

    def forward(self, x, t=None):
        """
        Forward pass.
        If t is None, assumes x has time embedding or is just feature extraction.
        If t is provided, embeds t and concatenates.
        """
        if t is None:
            if x.shape[1] == self.img_channels + 1:
                return self.denoiser(x)
            # Default to t=0? Or error?
            # For simplicity, assume t=0 (cleanest) if not provided?
            # Or just fail.
            raise ValueError("t must be provided for diffusion forward pass")

        return self.predict_x0(x, t)

    @torch.no_grad()
    def sample(self, num_samples=16, img_size=(1, 28, 28), device="cpu"):
        """
        Generate samples using the trained model via DDPM sampling.
        """
        self.eval()
        B = num_samples
        C, H, W = img_size

        # Start from pure noise
        x = torch.randn(B, C, H, W, device=device)

        for i in reversed(range(0, self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # 1. Predict x_0
            x_0_pred = self.predict_x0(x, t)

            # 2. Compute posterior mean for x_{t-1}
            # mu_t = coeff1 * x_0_pred + coeff2 * x_t

            # Extract coefficients
            alpha_t = self.alpha[t].view(B, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
            alpha_bar_prev_t = self.alpha_bar_prev[t].view(B, 1, 1, 1)
            beta_t = self.beta[t].view(B, 1, 1, 1)

            coeff1 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
            coeff2 = (
                torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
            )

            mean = coeff1 * x_0_pred + coeff2 * x

            # 3. Add noise
            if i > 0:
                noise = torch.randn_like(x)
                # Fixed variance sigma^2 = beta_t or posterior_variance
                # We use posterior_variance (sigma_tilde)
                var = self.posterior_variance[t].view(B, 1, 1, 1)
                sigma = torch.sqrt(var)
                x = mean + sigma * noise
            else:
                x = mean

        self.train()
        return x.clamp(
            -1, 1
        )  # Assume normalized to [-1, 1] usually, or [0,1] if data was [0,1]
