# =============================================================================
# Trainer — Multi-task training loop with early stopping & checkpointing
# =============================================================================
"""
Handles the full training lifecycle for the NetworkWorldModel:
  - Multi-task loss computation (state, infiltration, stage).
  - Gradient clipping.
  - Learning-rate scheduling.
  - Early stopping on validation loss.
  - Model checkpointing.
  - Training history recording.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.world_model import NetworkWorldModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Multi-task trainer for the GRU world model.

    Loss = α·L_state + β·L_infiltration + γ·L_stage
    """

    def __init__(
        self,
        model: NetworkWorldModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        tc = config.get("training", {})
        mc = config.get("model", {})

        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Loss weights
        self.alpha: float = mc.get("alpha", 1.0)
        self.beta: float = mc.get("beta", 1.0)
        self.gamma: float = mc.get("gamma", 1.0)

        # Loss functions
        self.state_loss_fn = nn.SmoothL1Loss()
        self.infiltration_loss_fn = nn.BCELoss()
        self.stage_loss_fn = nn.CrossEntropyLoss()

        # Optimiser
        lr = tc.get("learning_rate", 1e-3)
        wd = tc.get("weight_decay", 1e-4)
        opt_name = tc.get("optimizer", "adam").lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        sched = tc.get("scheduler", "cosine")
        epochs = tc.get("epochs", 100)
        if sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif sched == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=tc.get("scheduler_step_size", 20),
                gamma=tc.get("scheduler_gamma", 0.5),
            )
        elif sched == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        else:
            self.scheduler = None

        # Training params
        self.epochs: int = epochs
        self.batch_size: int = tc.get("batch_size", 64)
        self.gradient_clip: float = tc.get("gradient_clip", 1.0)
        self.patience: int = tc.get("early_stopping_patience", 15)
        self.min_delta: float = tc.get("min_delta", 1e-4)
        self.save_dir: str = tc.get("save_dir", "models/saved")
        self.checkpoint_dir: str = tc.get("checkpoint_dir", "models/checkpoints")

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_state_loss": [],
            "train_infil_loss": [],
            "train_stage_loss": [],
            "val_state_loss": [],
            "val_infil_loss": [],
            "val_stage_loss": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # DataLoader creation
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        X: np.ndarray,
        y_state: np.ndarray,
        y_attack: np.ndarray,
        y_binary: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Wrap numpy arrays in a PyTorch DataLoader."""
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_state, dtype=torch.float32),
            torch.tensor(y_attack, dtype=torch.long),
            torch.tensor(y_binary, dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        pred_state: torch.Tensor,
        pred_infil: torch.Tensor,
        pred_stage: torch.Tensor,
        y_state: torch.Tensor,
        y_attack: torch.Tensor,
        y_binary: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float, float]:
        """Compute the multi-task loss."""
        loss_state = self.state_loss_fn(pred_state, y_state)
        loss_infil = self.infiltration_loss_fn(pred_infil.squeeze(-1), y_binary)
        loss_stage = self.stage_loss_fn(pred_stage, y_attack)

        total = (
            self.alpha * loss_state
            + self.beta * loss_infil
            + self.gamma * loss_stage
        )
        return total, loss_state.item(), loss_infil.item(), loss_stage.item()

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float, float, float]:
        """Run one training epoch.  Returns (total, state, infil, stage) losses."""
        self.model.train()
        total_loss, s_loss, i_loss, g_loss, n = 0, 0, 0, 0, 0

        for batch in loader:
            X_b, ys_b, ya_b, yb_b = [t.to(self.device) for t in batch]

            self.optimizer.zero_grad()
            pred_state, pred_infil, pred_stage, _ = self.model(X_b)
            loss, sl, il, gl = self._compute_loss(
                pred_state, pred_infil, pred_stage, ys_b, ya_b, yb_b
            )
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()

            bs = X_b.size(0)
            total_loss += loss.item() * bs
            s_loss += sl * bs
            i_loss += il * bs
            g_loss += gl * bs
            n += bs

        return total_loss / n, s_loss / n, i_loss / n, g_loss / n

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float, float, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss, s_loss, i_loss, g_loss, n = 0, 0, 0, 0, 0

        for batch in loader:
            X_b, ys_b, ya_b, yb_b = [t.to(self.device) for t in batch]
            pred_state, pred_infil, pred_stage, _ = self.model(X_b)
            loss, sl, il, gl = self._compute_loss(
                pred_state, pred_infil, pred_stage, ys_b, ya_b, yb_b
            )
            bs = X_b.size(0)
            total_loss += loss.item() * bs
            s_loss += sl * bs
            i_loss += il * bs
            g_loss += gl * bs
            n += bs

        return total_loss / n, s_loss / n, i_loss / n, g_loss / n

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_state_train: np.ndarray,
        y_attack_train: np.ndarray,
        y_binary_train: np.ndarray,
        X_val: np.ndarray,
        y_state_val: np.ndarray,
        y_attack_val: np.ndarray,
        y_binary_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            X_train: (N_train, seq_len, D) training sequences.
            y_state_train: (N_train, D) next-state targets.
            y_attack_train: (N_train,) attack-stage targets.
            y_binary_train: (N_train,) infiltration targets.
            X_val / y_*_val: Corresponding validation arrays.

        Returns:
            Training history dictionary.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        train_loader = self._make_loader(
            X_train, y_state_train, y_attack_train, y_binary_train, shuffle=True
        )
        val_loader = self._make_loader(
            X_val, y_state_val, y_attack_val, y_binary_val, shuffle=False
        )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Starting training: %d epochs, batch_size=%d", self.epochs, self.batch_size)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_total, train_s, train_i, train_g = self._train_epoch(train_loader)
            val_total, val_s, val_i, val_g = self._val_epoch(val_loader)

            # Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_total)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_total)
            self.history["val_loss"].append(val_total)
            self.history["train_state_loss"].append(train_s)
            self.history["train_infil_loss"].append(train_i)
            self.history["train_stage_loss"].append(train_g)
            self.history["val_state_loss"].append(val_s)
            self.history["val_infil_loss"].append(val_i)
            self.history["val_stage_loss"].append(val_g)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | Train %.4f | Val %.4f | LR %.2e | %.1fs",
                epoch, self.epochs, train_total, val_total, current_lr, elapsed,
            )

            # ── Early stopping & checkpointing ──────────────────────
            if val_total < best_val_loss - self.min_delta:
                best_val_loss = val_total
                patience_counter = 0
                # Save best model
                best_path = os.path.join(self.save_dir, "best_world_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_total,
                    "config": self.config,
                }, best_path)
                logger.info("  [*] Best model saved (val_loss=%.4f)", val_total)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d).",
                        epoch, self.patience,
                    )
                    break

            # Periodic checkpoint
            if epoch % 10 == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_loss": val_total,
                }, ckpt_path)

        logger.info("Training complete.  Best val_loss: %.4f", best_val_loss)
        return self.history
