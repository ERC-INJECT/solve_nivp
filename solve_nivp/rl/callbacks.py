"""Callback utilities for RL training with adaptive stepping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:  # pragma: no cover - import guard
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Callback utilities require stable-baselines3. Install solve_nivp[rl] or add the dependency manually."
    ) from exc

try:  # pragma: no cover - optional plotting
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

__all__ = ["RewardCallback", "CustomMetricsCallback", "MetricsFigureConfig"]


class RewardCallback(BaseCallback):
    """Collect reward history during training for quick inspection."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.reward_history: List[float] = []
        self.timesteps: List[int] = []

    def _on_step(self) -> bool:  # noqa: D401
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.reward_history.append(rewards[0])
            self.timesteps.append(self.num_timesteps)
        return True


@dataclass(slots=True)
class MetricsFigureConfig:
    """Configuration for the metrics plot drawn at training end."""

    figsize: tuple[int, int] = (15, 8)
    filepath: str | None = "Images/training_metrics_FOSM.pdf"


class CustomMetricsCallback(BaseCallback):
    """Track reward, policy losses, and entropy coefficient over training."""

    def __init__(self, verbose: int = 0, *, fig_config: MetricsFigureConfig | None = None) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_timesteps: list[int] = []
        self.actor_losses: list[float] = []
        self.critic_losses: list[float] = []
        self.ent_coefs: list[float] = []
        self.metric_timesteps: list[int] = []
        self.current_ep_rewards: list[float] = []
        self.average_rewards: list[float] = []
        self.fig_config = fig_config or MetricsFigureConfig()

    def _on_training_start(self) -> None:  # noqa: D401
        self.current_ep_rewards = [0.0] * self.training_env.num_envs

    def _on_step(self) -> bool:  # noqa: D401
        rewards = self.locals.get("rewards", np.zeros(self.training_env.num_envs))
        dones = self.locals.get("dones", np.zeros(self.training_env.num_envs, dtype=bool))

        for env_idx in range(self.training_env.num_envs):
            self.current_ep_rewards[env_idx] += rewards[env_idx]
            if dones[env_idx]:
                self.episode_rewards.append(self.current_ep_rewards[env_idx])
                self.episode_timesteps.append(self.num_timesteps)
                self.current_ep_rewards[env_idx] = 0.0
                self.average_rewards.append(float(np.mean(self.episode_rewards)))

        if self.model is not None and hasattr(self.model, "logger"):
            logged_metrics = self.model.logger.name_to_value
            if "train/actor_loss" in logged_metrics:
                self.actor_losses.append(logged_metrics["train/actor_loss"])
                self.critic_losses.append(logged_metrics["train/critic_loss"])
                self.ent_coefs.append(logged_metrics["train/ent_coef"])
                self.metric_timesteps.append(self.num_timesteps)

        return True

    def on_training_end(self) -> None:  # noqa: D401
        if plt is None:
            if self.verbose > 0:
                print("matplotlib is not available; skipping metrics plot.")
            return

        plt.figure(figsize=self.fig_config.figsize)

        plt.subplot(2, 2, 1)
        episodes = np.arange(1, len(self.average_rewards) + 1)
        plt.plot(episodes, self.average_rewards, label="Average Reward")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Average Reward per Episode")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.metric_timesteps, self.actor_losses, label="Actor Loss")
        plt.plot(self.metric_timesteps, self.critic_losses, label="Critic Loss")
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(self.metric_timesteps, self.ent_coefs, label="Entropy Coef")
        plt.xlabel("Timesteps")
        plt.ylabel("Entropy Coefficient")
        plt.title("Entropy Coefficient")
        plt.grid(True)

        plt.tight_layout()
        if self.fig_config.filepath:
            plt.savefig(self.fig_config.filepath)
        plt.show()
        plt.close()
