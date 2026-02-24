"""
EquiTile RL: EquiTile for Reinforcement Learning
=================================================

Extends EquiTile with reinforcement learning capabilities:
- RLEquiTile: Policy and value networks for RL
- Actor-Critic architecture with tile-based learning
- Support for discrete and continuous action spaces
- Integration with Gymnasium environments

Examples
--------
>>> from bioplausible.models.equitile.rl import RLEquiTile, RLEquiTileConfig
>>> config = RLEquiTileConfig(
...     obs_dim=8,
...     action_dim=4,
...     action_type="discrete",
... )
>>> model = RLEquiTile(config)
>>> action, value, log_prob = model.act(observation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from bioplausible.models.base import BioModel, ModelConfig, register_model
from bioplausible.models.equitile.config import EquiTileConfig
from bioplausible.models.equitile.core import EquiTile

if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RLEquiTileConfig:
    """Configuration for RL EquiTile.

    Environment
    -----------
    obs_dim : int
        Observation dimension
    action_dim : int
        Action dimension (size of discrete space or dim of continuous)
    action_type : str
        Action space type: 'discrete' or 'continuous'

    Architecture
    ------------
    hidden_dim : int
        Hidden layer dimension
    num_layers : int
        Number of hidden layers
    neurons_per_tile : int
        Neurons per tile
    tiles_per_layer : int
        Tiles per layer

    Policy Settings
    ---------------
    log_std_init : float
        Initial log std for continuous actions
    log_std_min : float
        Minimum log std
    log_std_max : float
        Maximum log std

    Learning
    --------
    learning_rate : float
        Base learning rate
    entropy_coef : float
        Entropy regularization coefficient
    value_coef : float
        Value loss coefficient
    max_grad_norm : float
        Maximum gradient norm
    """

    # Environment
    obs_dim: int = 8
    action_dim: int = 4
    action_type: Literal["discrete", "continuous"] = "discrete"

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 2
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4

    # Policy settings
    log_std_init: float = 0.0
    log_std_min: float = -20
    log_std_max: float = 2

    # Learning
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # EquiTile settings
    mode: Literal["pc", "ep", "backprop"] = (
        "backprop"  # Default to backprop for RL stability
    )
    inference_steps: int = 5
    equitile_kwargs: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RL EquiTile Network
# =============================================================================


@register_model("rl_equitile")
class RLEquiTile(BioModel):
    """EquiTile for Reinforcement Learning.

    Implements actor-critic architecture with tile-based local learning
    for both policy and value functions.

    Parameters
    ----------
    config : RLEquiTileConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters

    Examples
    --------
    >>> config = RLEquiTileConfig(
    ...     obs_dim=8,
    ...     action_dim=4,
    ...     action_type="discrete",
    ... )
    >>> model = RLEquiTile(config)
    >>> action, value, log_prob = model.act(obs)
    """

    algorithm_name = "RLEquiTile"

    def __init__(
        self,
        config: Optional[RLEquiTileConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = RLEquiTileConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="rl_equitile",
                input_dim=config.obs_dim,
                output_dim=config.action_dim,
            )
        )

        self.config = config

        # Shared feature extractor (EquiTile-based)
        self.feature_extractor = self._build_feature_extractor(config)

        # Actor (policy) head
        self.actor = self._build_actor(config)

        # Critic (value) head
        self.critic = self._build_critic(config)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
        )

        # Initialize weights
        self._init_weights()

    def _build_feature_extractor(self, config: RLEquiTileConfig) -> nn.Module:
        """Build EquiTile feature extractor.

        Parameters
        ----------
        config : RLEquiTileConfig
            Configuration

        Returns
        -------
        nn.Module
            Feature extractor
        """
        # Use EquiTile as feature extractor
        # Output dimension matches the expected input for actor/critic heads
        tile_dim = config.neurons_per_tile * config.tiles_per_layer

        equitile_config = EquiTileConfig(
            neurons_per_tile=config.neurons_per_tile,
            num_layers=config.num_layers,
            tiles_per_layer=config.tiles_per_layer,
            mode=config.mode,
            inference_steps=config.inference_steps,
            learning_rate=config.learning_rate,
            **config.equitile_kwargs,
        )

        return EquiTile(
            config=equitile_config,
            input_dim=config.obs_dim,
            output_dim=tile_dim,  # Features for heads
        )

    def _build_actor(self, config: RLEquiTileConfig) -> nn.Module:
        """Build actor (policy) head.

        Parameters
        ----------
        config : RLEquiTileConfig
            Configuration

        Returns
        -------
        nn.Module
            Actor head
        """
        tile_dim = config.neurons_per_tile * config.tiles_per_layer

        if config.action_type == "discrete":
            return nn.Linear(tile_dim, config.action_dim)
        else:
            # Continuous action space
            actor = nn.Linear(tile_dim, config.action_dim)
            # Initialize to small values for stable training
            nn.init.orthogonal_(actor.weight, gain=0.01)
            nn.init.zeros_(actor.bias)
            return actor

    def _build_critic(self, config: RLEquiTileConfig) -> nn.Module:
        """Build critic (value) head.

        Parameters
        ----------
        config : RLEquiTileConfig
            Configuration

        Returns
        -------
        nn.Module
            Critic head
        """
        tile_dim = config.neurons_per_tile * config.tiles_per_layer
        critic = nn.Linear(tile_dim, 1)
        nn.init.orthogonal_(critic.weight, gain=1.0)
        nn.init.zeros_(critic.bias)
        return critic

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(
                        module.weight, gain=nn.init.calculate_gain("relu")
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def extract_features(self, obs: Tensor) -> Tensor:
        """Extract features from observation.

        Parameters
        ----------
        obs : torch.Tensor
            Observation

        Returns
        -------
        torch.Tensor
            Features
        """
        return self.feature_extractor(obs)

    def act(
        self,
        obs: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select action.

        Parameters
        ----------
        obs : torch.Tensor
            Observation (batch, obs_dim)
        deterministic : bool
            If True, use deterministic action selection

        Returns
        -------
        tuple
            (action, value, log_prob)
        """
        # Extract features
        features = self.extract_features(obs)

        # Get policy logits/parameters
        if self.config.action_type == "discrete":
            action_logits = self.actor(features)
            dist = Categorical(logits=action_logits)
        else:
            action_mean = self.actor(features)
            action_log_std = torch.clamp(
                torch.ones_like(action_mean) * self.config.log_std_init,
                self.config.log_std_min,
                self.config.log_std_max,
            )
            action_std = torch.exp(action_log_std)
            dist = Normal(action_mean, action_std)

        # Sample action
        if deterministic:
            if self.config.action_type == "discrete":
                action = action_logits.argmax(dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()

        # Get value
        value = self.critic(features).squeeze(-1)

        # Get log probability
        if self.config.action_type == "discrete":
            log_prob = dist.log_prob(action).unsqueeze(-1)
        else:
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, value, log_prob

    def evaluate_actions(
        self,
        obs: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions for PPO-style updates.

        Parameters
        ----------
        obs : torch.Tensor
            Observations
        actions : torch.Tensor
            Actions taken

        Returns
        -------
        tuple
            (log_prob, entropy, value)
        """
        # Extract features
        features = self.extract_features(obs)

        # Get distribution
        if self.config.action_type == "discrete":
            action_logits = self.actor(features)
            dist = Categorical(logits=action_logits)
            # For discrete, log_prob and entropy are already per-sample
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            action_mean = self.actor(features)
            action_log_std = torch.clamp(
                torch.ones_like(action_mean) * self.config.log_std_init,
                self.config.log_std_min,
                self.config.log_std_max,
            )
            action_std = torch.exp(action_log_std)
            dist = Normal(action_mean, action_std)
            # For continuous, sum over action dimension
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

        value = self.critic(features).squeeze(-1)

        return log_prob, entropy, value

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate.

        Parameters
        ----------
        obs : torch.Tensor
            Observation

        Returns
        -------
        torch.Tensor
            Value estimate
        """
        features = self.extract_features(obs)
        return self.critic(features).squeeze(-1)

    def forward(
        self,
        obs: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass (act).

        Parameters
        ----------
        obs : torch.Tensor
            Observation
        deterministic : bool
            If True, use deterministic action selection

        Returns
        -------
        tuple
            (action, value, log_prob)
        """
        return self.act(obs, deterministic)

    def compute_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute PPO-style loss.

        Parameters
        ----------
        obs : torch.Tensor
            Observations
        actions : torch.Tensor
            Actions taken
        advantages : torch.Tensor
            Advantage estimates
        returns : torch.Tensor
            Return estimates
        old_log_probs : torch.Tensor
            Log probabilities from old policy

        Returns
        -------
        dict
            Loss components
        """
        # Evaluate actions
        log_prob, entropy, value = self.evaluate_actions(obs, actions)

        # Policy loss (PPO clip)
        ratio = torch.exp(log_prob - old_log_probs)
        clip_ratio = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(value, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy.mean(),
            "ratio": ratio.mean(),
        }

    def train_step(
        self,
        obs: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
    ) -> Dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        obs : torch.Tensor
            Observations
        actions : torch.Tensor
            Actions taken
        advantages : torch.Tensor
            Advantage estimates
        returns : torch.Tensor
            Return estimates
        old_log_probs : torch.Tensor
            Log probabilities from old policy

        Returns
        -------
        dict
            Training statistics
        """
        # Compute loss
        loss_dict = self.compute_loss(obs, actions, advantages, returns, old_log_probs)

        # Backward pass
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)

        # Update
        self.optimizer.step()

        # Return scalar values
        return {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in loss_dict.items()
        }


# =============================================================================
# Recurrent RL EquiTile (for Partial Observability)
# =============================================================================


class RecurrentRLEquiTile(RLEquiTile):
    """Recurrent EquiTile for partially observable environments.

    Adds LSTM/GRU layers for temporal memory.

    Parameters
    ----------
    config : RLEquiTileConfig
        Configuration
    rnn_type : str
        RNN type: 'lstm' or 'gru'
    rnn_hidden_dim : int
        RNN hidden dimension
    """

    def __init__(
        self,
        config: RLEquiTileConfig,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        rnn_hidden_dim: int = 128,
    ) -> None:
        super().__init__(config)

        self.rnn_hidden_dim = rnn_hidden_dim

        # Replace feature extractor with recurrent version
        tile_dim = config.neurons_per_tile * config.tiles_per_layer

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(tile_dim, rnn_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(tile_dim, rnn_hidden_dim, batch_first=True)

        # Update actor and critic for rnn output
        self.actor = nn.Linear(rnn_hidden_dim, config.action_dim)
        self.critic = nn.Linear(rnn_hidden_dim, 1)

        # Hidden state
        self._hidden_state = None

    def reset_hidden(self, batch_size: int, device: torch.device) -> None:
        """Reset hidden state.

        Parameters
        ----------
        batch_size : int
            Batch size
        device : torch.device
            Device
        """
        self._hidden_state = (
            torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device),
            torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device),
        )

    def extract_features(self, obs: Tensor) -> Tensor:
        """Extract features with recurrence.

        Parameters
        ----------
        obs : torch.Tensor
            Observation (batch, obs_dim) or (batch, seq_len, obs_dim)

        Returns
        -------
        torch.Tensor
            Features
        """
        # Get base features
        base_features = self.feature_extractor(obs)

        # Handle sequence vs single step
        if base_features.dim() == 2:
            base_features = base_features.unsqueeze(1)

        # Run through RNN
        if self._hidden_state is not None:
            output, self._hidden_state = self.rnn(base_features, self._hidden_state)
        else:
            output, _ = self.rnn(base_features)

        return output.squeeze(1)


# =============================================================================
# RL Utilities
# =============================================================================


class RolloutBuffer:
    """Rollout buffer for on-policy RL algorithms.

    Stores trajectories for PPO-style updates.

    Parameters
    ----------
    obs_dim : int
        Observation dimension
    action_dim : int
        Action dimension
    device : torch.device
        Device
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs: List[Tensor] = []
        self.actions: List[Tensor] = []
        self.rewards: List[Tensor] = []
        self.dones: List[Tensor] = []
        self.values: List[Tensor] = []
        self.log_probs: List[Tensor] = []

    def add(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        value: Tensor,
        log_prob: Tensor,
    ) -> None:
        """Add transition to buffer.

        Parameters
        ----------
        obs : torch.Tensor
            Observation
        action : torch.Tensor
            Action
        reward : torch.Tensor
            Reward
        done : torch.Tensor
            Done flag
        value : torch.Tensor
            Value estimate
        log_prob : torch.Tensor
            Log probability
        """
        self.obs.append(obs.clone())
        self.actions.append(action.clone())
        self.rewards.append(reward.clone())
        self.dones.append(done.clone())
        self.values.append(value.clone())
        self.log_probs.append(log_prob.clone())

    def get(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        last_value: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get buffered data with GAE advantages.

        Parameters
        ----------
        gamma : float
            Discount factor
        lam : float
            GAE lambda
        last_value : float
            Value of final state

        Returns
        -------
        tuple
            (obs, actions, advantages, returns, old_log_probs)
        """
        # Stack buffers
        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)

        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            lam=lam,
            last_value=last_value,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clear buffer
        self.clear()

        return obs, actions, advantages, returns, log_probs

    def clear(self) -> None:
        """Clear buffer."""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.obs)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : torch.Tensor
        Rewards
    values : torch.Tensor
        Value estimates
    dones : torch.Tensor
        Done flags
    gamma : float
        Discount factor
    lam : float
        GAE lambda
    last_value : float
        Value of final state

    Returns
    -------
    tuple
        (advantages, returns)
    """
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages)
    returns = advantages + values

    return advantages, returns


# =============================================================================
# Factory Functions
# =============================================================================


def create_rl_model(
    obs_dim: int,
    action_dim: int,
    action_type: Literal["discrete", "continuous"] = "discrete",
    hidden_dim: int = 128,
    **kwargs: Any,
) -> RLEquiTile:
    """Create RLEquiTile model.

    Parameters
    ----------
    obs_dim : int
        Observation dimension
    action_dim : int
        Action dimension
    action_type : str
        Action space type
    hidden_dim : int
        Hidden dimension
    **kwargs
        Additional arguments

    Returns
    -------
    RLEquiTile
        RL model
    """
    config = RLEquiTileConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    return RLEquiTile(config)


def create_recurrent_rl_model(
    obs_dim: int,
    action_dim: int,
    action_type: Literal["discrete", "continuous"] = "discrete",
    rnn_hidden_dim: int = 128,
    **kwargs: Any,
) -> RecurrentRLEquiTile:
    """Create RecurrentRLEquiTile model.

    Parameters
    ----------
    obs_dim : int
        Observation dimension
    action_dim : int
        Action dimension
    action_type : str
        Action space type
    rnn_hidden_dim : int
        RNN hidden dimension
    **kwargs
        Additional arguments

    Returns
    -------
    RecurrentRLEquiTile
        Recurrent RL model
    """
    config = RLEquiTileConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        **kwargs,
    )
    return RecurrentRLEquiTile(config, rnn_hidden_dim=rnn_hidden_dim)


def create_atari_model(
    obs_shape: Tuple[int, int, int] = (4, 84, 84),
    action_dim: int = 4,
    **kwargs: Any,
) -> RLEquiTile:
    """Create RLEquiTile for Atari games.

    Note: Flattens the image observation to a 1D vector.

    Parameters
    ----------
    obs_shape : tuple
        Observation shape (channels, height, width)
    action_dim : int
        Number of actions
    **kwargs
        Additional arguments

    Returns
    -------
    RLEquiTile
        Atari model (MLP-based)
    """
    # Flatten image observation
    obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]

    return create_rl_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=512,
        **kwargs,
    )


def create_mujoco_model(
    obs_dim: int,
    action_dim: int,
    **kwargs: Any,
) -> RLEquiTile:
    """Create RLEquiTile for MuJoCo environments (continuous action space).

    Parameters
    ----------
    obs_dim : int
        Observation dimension
    action_dim : int
        Action dimension
    **kwargs
        Additional arguments

    Returns
    -------
    RLEquiTile
        MuJoCo model
    """
    return create_rl_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type="continuous",
        hidden_dim=256,
        **kwargs,
    )
