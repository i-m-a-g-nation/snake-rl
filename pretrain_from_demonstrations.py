"""从示范数据预训练 DQN。"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import DuelingDQN
from utils import ensure_dir, append_train_log


def pretrain(args):
    """从示范数据预训练。"""
    ensure_dir(os.path.dirname(args.save_path))
    ensure_dir(os.path.dirname(args.log_path))

    # 加载示范数据
    print(f"加载示范数据: {args.demo}")
    data = np.load(args.demo, allow_pickle=True)
    states = data["states"]
    actions = data["actions"]

    print(f"  样本数: {len(states)}")
    print(f"  状态维度: {states.shape[1] if len(states.shape) > 1 else 'N/A'}")
    print(f"  动作维度: {len(np.unique(actions))}")

    # 创建模型
    state_dim = states.shape[1]
    action_dim = len(np.unique(actions))

    model = DuelingDQN(state_dim, action_dim, hidden_dim=128)

    # 如果有预训练模型，加载
    if args.load_model and os.path.exists(args.load_model):
        print(f"加载预训练模型: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location="cpu")
        model.load_state_dict(checkpoint["policy_net"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"设备: {device}")

    # 准备数据
    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    dataset = TensorDataset(states_t, actions_t)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练
    print(f"\n开始预训练...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print("-" * 60)

    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            q_values = model(batch_states)
            loss = criterion(q_values, batch_actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_states)
            predictions = q_values.argmax(dim=1)
            correct += (predictions == batch_actions).sum().item()
            total += len(batch_states)

        avg_loss = total_loss / total
        accuracy = correct / total * 100

        log_record = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "accuracy": round(accuracy, 2),
        }
        log_records.append(log_record)
        append_train_log(args.log_path, log_record)

        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%")

    # 保存模型
    torch.save({
        "policy_net": model.state_dict(),
        "target_net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_step_count": 0,
    }, args.save_path)

    print(f"\n预训练完成!")
    print(f"  最终 Loss: {avg_loss:.6f}")
    print(f"  最终 Accuracy: {accuracy:.2f}%")
    print(f"  模型已保存: {args.save_path}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从示范数据预训练 DQN")
    parser.add_argument("--demo", type=str, required=True, help="示范数据路径")
    parser.add_argument("--state-mode", type=str, default="basic17", help="状态模式")
    parser.add_argument("--dueling", action="store_true", default=True, help="使用 Dueling DQN")
    parser.add_argument("--load-model", type=str, default=None, help="加载已有模型继续预训练")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--save-path", type=str, default="checkpoints/dqfd_lite/pretrained.pt", help="保存路径")
    parser.add_argument("--log-path", type=str, default="logs/dqfd_lite/pretrain_log.csv", help="日志路径")
    args = parser.parse_args()

    pretrain(args)
