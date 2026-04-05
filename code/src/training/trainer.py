import os
import json
from datetime import datetime

import torch
import torch.nn as nn


class Trainer:
    """训练循环，包含验证、早停、学习率调度、checkpoint 保存、日志记录。"""

    def __init__(self, model, criterion, optimizer, scheduler, device,
                 patience=12, checkpoint_path=None, log_dir="logs"):
        self.model = model
        self.criterion = criterion
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
        # 初始化实时日志：logs/run_时间戳/
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "train.log")
        self._log_line("Epoch | Train NLL | Val NLL | Patience | LR")

        for epoch in range(num_epochs):
            # ---- 训练 ----
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

            # ---- 验证 ----
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

            # 学习率调度
            self.scheduler.step(avg_val)

            # 早停检查
            if avg_val < self.best_val_loss:
                self.best_val_loss = avg_val
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                if self.checkpoint_path:
                    # 按实验运行保存：在原路径中插入 run_id
                    base, ext = os.path.splitext(self.checkpoint_path)
                    ckpt = f"{base}_{self.run_id}{ext}"
                    torch.save(self.best_model_state, ckpt)
            else:
                self.patience_counter += 1

            # 实时写入日志（每个 epoch 都写）
            lr = self.optimizer.param_groups[0]["lr"]
            self._log_line(
                f"{epoch+1:3d}/{num_epochs} | {avg_train:.6f} | {avg_val:.6f} | "
                f"{self.patience_counter}/{self.patience} | {lr:.2e}"
            )

            # 打印日志（部分 epoch）
            if (epoch + 1) % 5 == 0 or self.patience_counter == 0:
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train NLL: {avg_train:.6f} | "
                    f"Val NLL: {avg_val:.6f} | "
                    f"Patience: {self.patience_counter}/{self.patience}"
                )

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at Epoch {epoch+1}")
                break

        # 恢复最优权重
        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        print("Best model weights loaded")

    def _log_line(self, line: str):
        """实时追加一行到日志文件。"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_log(self, metrics: dict = None, config: dict = None):
        """将训练日志和评估指标保存到 logs/ 目录。"""
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
