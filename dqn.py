import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minatar import Environment
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="breakout")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frames", type=int, default=5_000_000)
    p.add_argument("--replay-size", type=int, default=100_000)
    p.add_argument("--replay-start", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--target-update", type=int, default=1_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay-frames", type=int, default=100_000)
    p.add_argument("--double", action="store_true")
    p.add_argument("--dueling", action="store_true")
    p.add_argument("--per", action="store_true")
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta", type=float, default=0.4)
    p.add_argument("--c51", action="store_true")
    p.add_argument("--num-atoms", type=int, default=51)
    p.add_argument("--v-min", type=float, default=0.0)
    p.add_argument("--v-max", type=float, default=100.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-name", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt-freq", type=int, default=250_000)
    return p.parse_args()


class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        conv_out = 8 * 8 * 16
        self.fc = nn.Linear(conv_out, 128)
        if dueling:
            self.value = nn.Linear(128, 1)
            self.advantage = nn.Linear(128, num_actions)
        else:
            self.head = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc(x.reshape(x.size(0), -1)))
        if self.dueling:
            v = self.value(x)
            a = self.advantage(x)
            return v + a - a.mean(1, keepdim=True)
        return self.head(x)


class CategoricalQNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, num_atoms):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        conv_out = 8 * 8 * 16
        self.fc = nn.Linear(conv_out, 128)
        self.head = nn.Linear(128, num_actions * num_atoms)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc(x.reshape(x.size(0), -1)))
        return self.head(x).view(-1, self.num_actions, self.num_atoms)


class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, ns, d):
        i = self.pos
        self.states[i] = s
        self.next_states[i] = ns
        self.actions[i] = a
        self.rewards[i] = r
        self.dones[i] = d
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx].astype(np.float32),
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx].astype(np.float32),
            self.dones[idx],
        )


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    def total(self):
        return self.tree[0]

    def update(self, leaf, priority):
        i = leaf + self.capacity - 1
        change = priority - self.tree[i]
        self.tree[i] = priority
        while i > 0:
            i = (i - 1) // 2
            self.tree[i] += change

    def sample_leaf(self, value):
        i = 0
        while 2 * i + 1 < len(self.tree):
            left = 2 * i + 1
            if value <= self.tree[left]:
                i = left
            else:
                value -= self.tree[left]
                i = left + 1
        return i - (self.capacity - 1)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, ns, d):
        i = self.pos
        self.states[i] = s
        self.next_states[i] = ns
        self.actions[i] = a
        self.rewards[i] = r
        self.dones[i] = d
        self.tree.update(i, self.max_priority ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        total = self.tree.total()
        segment = total / batch_size
        idx = np.zeros(batch_size, dtype=np.int64)
        for j in range(batch_size):
            value = min(np.random.uniform(segment * j, segment * (j + 1)), total)
            idx[j] = self.tree.sample_leaf(value)
        probs = self.tree.tree[idx + self.capacity - 1] / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()
        return (self.states[idx].astype(np.float32), self.actions[idx], self.rewards[idx],
                self.next_states[idx].astype(np.float32), self.dones[idx], idx, weights.astype(np.float32))

    def update_priorities(self, idx, priorities):
        for i, p in zip(idx, priorities):
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(i), p ** self.alpha)


def get_state(env):
    # MinAtar returns (H, W, C) bool; conv wants (C, H, W) float
    return np.transpose(env.state(), (2, 0, 1)).astype(np.float32)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    parts = []
    if args.dueling:
        parts.append("dueling")
    if args.double:
        parts.append("double")
    if args.per:
        parts.append("per")
    if args.c51:
        parts.append("c51")
    parts.append("dqn")
    algo = "_".join(parts)
    run_name = args.log_name or f"{algo}_{args.game}_seed{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")

    env = Environment(args.game)
    env.random = np.random.RandomState(args.seed)
    env.env.random = np.random.RandomState(args.seed + 10_000)
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    if args.c51:
        q_net = CategoricalQNetwork(in_channels, num_actions, args.num_atoms).to(device)
        target_net = CategoricalQNetwork(in_channels, num_actions, args.num_atoms).to(device)
        support = torch.linspace(args.v_min, args.v_max, args.num_atoms).to(device)
        delta_z = (args.v_max - args.v_min) / (args.num_atoms - 1)
    else:
        q_net = QNetwork(in_channels, num_actions, args.dueling).to(device)
        target_net = QNetwork(in_channels, num_actions, args.dueling).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.RMSprop(q_net.parameters(), lr=args.lr, alpha=0.95, centered=True, eps=0.01)

    if args.per:
        buffer = PrioritizedReplayBuffer(args.replay_size, (in_channels, 10, 10), args.per_alpha)
    else:
        buffer = ReplayBuffer(args.replay_size, (in_channels, 10, 10))

    env.reset()
    state = get_state(env)
    episode_return = 0.0
    returns = deque(maxlen=100)
    episode = 0
    start_frame = 1
    ckpt_path = f"runs/{run_name}/ckpt.pt"
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        q_net.load_state_dict(ck["q_net"])
        target_net.load_state_dict(ck["target_net"])
        optimizer.load_state_dict(ck["optimizer"])
        returns = deque(ck["returns"], maxlen=100)
        episode = ck["episode"]
        start_frame = ck["frame"] + 1
        print(f"resumed from frame {ck['frame']}")
    start = time.time()

    for frame in range(start_frame, args.frames + 1):
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * frame / args.eps_decay_frames)

        if random.random() < eps:
            action = random.randrange(num_actions)
        else:
            with torch.no_grad():
                st = torch.from_numpy(state).unsqueeze(0).to(device)
                if args.c51:
                    q = (F.softmax(q_net(st), dim=2) * support).sum(2)
                else:
                    q = q_net(st)
                action = int(q.argmax(1).item())

        reward, done = env.act(action)
        next_state = get_state(env)
        buffer.add(state, action, reward, next_state, float(done))
        episode_return += reward
        state = next_state

        if done:
            returns.append(episode_return)
            writer.add_scalar("charts/episodic_return", episode_return, frame)
            if returns:
                writer.add_scalar("charts/return_avg100", np.mean(returns), frame)
            episode += 1
            env.reset()
            state = get_state(env)
            episode_return = 0.0

        if buffer.size >= args.replay_start and frame % args.train_freq == 0:
            if args.per:
                beta = min(1.0, args.per_beta + (1.0 - args.per_beta) * frame / args.frames)
                s, a, r, ns, d, idx, w = buffer.sample(args.batch_size, beta)
            else:
                s, a, r, ns, d = buffer.sample(args.batch_size)
            s = torch.from_numpy(s).to(device)
            a = torch.from_numpy(a).to(device)
            r = torch.from_numpy(r).to(device)
            ns = torch.from_numpy(ns).to(device)
            d = torch.from_numpy(d).to(device)

            bs = args.batch_size
            if args.c51:
                with torch.no_grad():
                    next_probs = F.softmax(target_net(ns), dim=2)
                    if args.double:
                        next_a = (F.softmax(q_net(ns), dim=2) * support).sum(2).argmax(1)
                    else:
                        next_a = (next_probs * support).sum(2).argmax(1)
                    next_dist = next_probs[torch.arange(bs), next_a]
                    Tz = r.unsqueeze(1) + args.gamma * (1.0 - d).unsqueeze(1) * support.unsqueeze(0)
                    Tz = Tz.clamp(args.v_min, args.v_max)
                    b = (Tz - args.v_min) / delta_z
                    lo = b.floor().long()
                    hi = b.ceil().long()
                    lo[(hi > 0) & (lo == hi)] -= 1
                    hi[(lo < args.num_atoms - 1) & (lo == hi)] += 1
                    m = torch.zeros(bs, args.num_atoms, device=device)
                    offset = (torch.arange(bs, device=device) * args.num_atoms).unsqueeze(1)
                    m.view(-1).index_add_(0, (lo + offset).view(-1), (next_dist * (hi.float() - b)).view(-1))
                    m.view(-1).index_add_(0, (hi + offset).view(-1), (next_dist * (b - lo.float())).view(-1))
                log_p = F.log_softmax(q_net(s), dim=2)[torch.arange(bs), a]
                loss_each = -(m * log_p).sum(1)
                if args.per:
                    priority = loss_each.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    if args.double:
                        next_actions = q_net(ns).argmax(1, keepdim=True)
                        target_q = target_net(ns).gather(1, next_actions).squeeze(1)
                    else:
                        target_q = target_net(ns).max(1)[0]
                    y = r + args.gamma * (1.0 - d) * target_q
                q_pred = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss_each = F.smooth_l1_loss(q_pred, y, reduction="none")
                if args.per:
                    priority = (y - q_pred).detach().abs().cpu().numpy()

            if args.per:
                w_t = torch.from_numpy(w).to(device)
                loss = (w_t * loss_each).mean()
            else:
                loss = loss_each.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.per:
                buffer.update_priorities(idx, priority + 1e-6)

            if frame % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

        if frame % 10_000 == 0:
            fps = int((frame - start_frame + 1) / (time.time() - start))
            avg = np.mean(returns) if returns else 0.0
            print(f"frame {frame} ep {episode} eps {eps:.3f} avg100 {avg:.2f} fps {fps}")
            writer.add_scalar("charts/eps", eps, frame)
            writer.add_scalar("charts/fps", fps, frame)

        if frame % args.ckpt_freq == 0:
            tmp = ckpt_path + ".tmp"
            torch.save({"q_net": q_net.state_dict(), "target_net": target_net.state_dict(),
                        "optimizer": optimizer.state_dict(), "returns": list(returns),
                        "episode": episode, "frame": frame}, tmp)
            os.replace(tmp, ckpt_path)

    torch.save(q_net.state_dict(), f"runs/{run_name}/model.pt")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    writer.close()


if __name__ == "__main__":
    main()
