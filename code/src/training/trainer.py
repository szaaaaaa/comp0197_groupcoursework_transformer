import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """Training loop with validation, early stopping, LR scheduling, and logging.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    criterion : nn.Module
        Loss function (called as ``criterion(mu, target, var)``).
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning-rate scheduler (``ReduceLROnPlateau``).
    device : torch.device
        Training device.
    patience : int
        Early-stopping patience (epochs without improvement).
    checkpoint_path : str or None
        Path template for saving the best checkpoint.
    log_dir : str
        Root directory for training logs.
    loss_name : str
        Display name for the loss (used in log headers).
    """

    def __init__(self, model, criterion, optimizer, scheduler, device,
                 patience=12, checkpoint_path=None, log_dir="logs", loss_name="NLL"):
        self.model = model
        self.criterion = criterion
        self.loss_name = loss_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.log_dir = log_dir

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

    def train(self, train_loader, val_loader, num_epochs):
        """Run the training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        num_epochs : int
            Maximum number of epochs.
        """
        # Initialize live log: logs/run_<timestamp>/
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "train.log")
        self._log_line(f"Epoch | Train {self.loss_name} | Val {self.loss_name} | Patience | LR")

        pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            # ---- Train ----
            self.model.train()
            total_loss, n_batches = 0.0, 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                mu, var = self.model(X_batch)
                loss = self.criterion(mu, y_batch, var)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            avg_train = total_loss / n_batches

            # ---- Validation ----
            self.model.eval()
            total_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    mu, var = self.model(X_batch)
                    loss = self.criterion(mu, y_batch, var)
                    total_loss += loss.item()
                    n_batches += 1
            avg_val = total_loss / n_batches

            self.train_losses.append(avg_train)
            self.val_losses.append(avg_val)

            # LR scheduling
            self.scheduler.step(avg_val)

            # Early stopping check
            star = ""
            if avg_val < self.best_val_loss:
                self.best_val_loss = avg_val
                self.patience_counter = 0
                star = " *"
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                if self.checkpoint_path:
                    base, ext = os.path.splitext(self.checkpoint_path)
                    ckpt = f"{base}_{self.run_id}{ext}"
                    torch.save(self.best_model_state, ckpt)
            else:
                self.patience_counter += 1

            # Update progress bar
            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix_str(
                f"train={avg_train:.4f} val={avg_val:.4f} "
                f"p={self.patience_counter}/{self.patience} lr={lr:.1e}{star}"
            )

            # Write log in real time
            self._log_line(
                f"{epoch+1:3d}/{num_epochs} | {avg_train:.6f} | {avg_val:.6f} | "
                f"{self.patience_counter}/{self.patience} | {lr:.2e}"
            )

            if self.patience_counter >= self.patience:
                pbar.close()
                print(f"Early stopping at Epoch {epoch+1}")
                break

        # Restore best weights
        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        print("Best model weights loaded")

    def _log_line(self, line: str):
        """Append one line to the log file.

        Parameters
        ----------
        line : str
            Text to write.
        """
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_log(self, metrics: dict = None, config: dict = None):
        """Save training log and evaluation metrics to the run directory.

        Parameters
        ----------
        metrics : dict, optional
            Evaluation metrics to include.
        config : dict, optional
            Experiment configuration to include.
        """
        log = {
            "timestamp": self.run_id,
            "total_epochs": len(self.train_losses),
            "best_val_loss": self.best_val_loss,
            "final_lr": self.optimizer.param_groups[0]["lr"],
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        if metrics:
            log["metrics"] = metrics
        if config:
            log["config"] = config
        path = os.path.join(self.run_dir, "result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"Log saved to {path}")
