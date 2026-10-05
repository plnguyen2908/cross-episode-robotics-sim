"""Minimal behavior cloning on an exported dataset, as a template.

Trains a small CNN + MLP that maps (wrist image, robot state) to the next
absolute action, then can be evaluated with scripts/run_policy.py through
`BCPolicy`. It shows the data path end to end; it is not a tuned baseline.
For serious training, point robomimic or LeRobot at the same HDF5 file.

    python examples/train_bc.py --dataset datasets/breakfast.hdf5 --output checkpoints/bc.pt
    python scripts/run_policy.py --task breakfast --policy examples.train_bc:BCPolicy \
        --policy-arg checkpoint=checkpoints/bc.pt --output runs/bc_eval
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGE_KEY = "wrist_cam_image"


class Demos(Dataset):
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        self.index = [(demo, t) for demo in self.file["data"] for t in range(self.file["data"][demo].attrs["num_samples"])]
        actions = np.concatenate([self.file["data"][d]["actions"][:] for d in self.file["data"]])
        states = np.concatenate([self.file["data"][d]["obs/state"][:] for d in self.file["data"]])
        self.stats = {k: (v.mean(0), v.std(0) + 1e-6) for k, v in (("action", actions), ("state", states))}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        demo, t = self.index[i]
        group = self.file["data"][demo]
        image = torch.from_numpy(group["obs"][IMAGE_KEY][t]).permute(2, 0, 1).float() / 255
        state = (group["obs/state"][t] - self.stats["state"][0]) / self.stats["state"][1]
        action = (group["actions"][t] - self.stats["action"][0]) / self.stats["action"][1]
        return image, torch.from_numpy(state).float(), torch.from_numpy(action).float()


class Net(nn.Module):
    def __init__(self, state_dim=11, action_dim=11):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2), nn.ReLU(), nn.Conv2d(32, 64, 5, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 + state_dim, 256), nn.ReLU(), nn.Linear(256, action_dim))

    def forward(self, image, state):
        return self.head(torch.cat([self.vision(image), state], dim=-1))


def train(dataset, output, epochs, batch_size, device):
    data = Demos(dataset)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
    net = Net().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    for epoch in range(epochs):
        total = 0.0
        for image, state, action in loader:
            loss = nn.functional.mse_loss(net(image.to(device), state.to(device)), action.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(image)
        print(f"epoch {epoch}: loss {total / len(data):.4f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=net.state_dict(), stats={k: [a.tolist() for a in v] for k, v in data.stats.items()}), output)


class BCPolicy:
    """Wraps a checkpoint for scripts/run_policy.py (needs the wrist camera in the observation)."""

    def __init__(self, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"):
        saved = torch.load(checkpoint, map_location=device)
        self.net = Net().to(device).eval()
        self.net.load_state_dict(saved["model"])
        self.stats = {k: [np.asarray(a) for a in v] for k, v in saved["stats"].items()}
        self.device = device

    @torch.no_grad()
    def act(self, observation):
        image = torch.from_numpy(observation["robot_0/wrist_cam"]).permute(2, 0, 1).float()[None] / 255
        mean, std = self.stats["state"]
        state = torch.from_numpy((observation["state"] - mean) / std).float()[None]
        action = self.net(image.to(self.device), state.to(self.device))[0].cpu().numpy()
        mean, std = self.stats["action"]
        return action * std + mean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.dataset, args.output, args.epochs, args.batch_size, args.device)


if __name__ == "__main__":
    main()
