import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


class QNetwork(nn.Module):
    def __init__(self, env: gym.vector.SyncVectorEnv) -> None:
        super().__init__()
        observation_size = int(np.prod(env.single_observation_space.shape))
        action_count = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(observation_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def linear_schedule(
    start_epsilon: float,
    end_epsilon: float,
    duration: int,
    timestep: int,
) -> float:
    progress = min(timestep / duration, 1.0)
    return start_epsilon + progress * (end_epsilon - start_epsilon)


def make_env() -> gym.Env:
    env = gym.make("CartPole-v1")
    return gym.wrappers.RecordEpisodeStatistics(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CleanRL-style DQN on CartPole.")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-frequency", type=int, default=10)
    parser.add_argument("--target-network-frequency", type=int, default=500)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--cuda", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    run_name = f"CartPole-v1__cleanrl_dqn__{args.seed}__{int(time.time())}"
    log_dir = Path("cleanrl_dqn_cartpole_tensorboard") / run_name
    writer = SummaryWriter(log_dir)

    envs = gym.vector.SyncVectorEnv([make_env])
    envs.single_action_space.seed(args.seed)
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("DQN requires a discrete action space.")

    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    observations, _ = envs.reset(seed=args.seed)
    start_time = time.time()
    exploration_steps = max(int(args.exploration_fraction * args.total_timesteps), 1)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_epsilon,
            args.end_epsilon,
            exploration_steps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(observations, device=device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_observations, rewards, terminations, truncations, infos = envs.step(
            actions
        )

        real_next_observations = next_observations.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                real_next_observations[index] = infos["final_observation"][index]

        replay_buffer.add(
            observations,
            real_next_observations,
            actions,
            rewards,
            terminations,
            infos,
        )
        observations = next_observations

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_return = float(info["episode"]["r"])
                    episode_length = int(info["episode"]["l"])
                    print(
                        f"step={global_step}, return={episode_return:.1f}, "
                        f"length={episode_length}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", episode_return, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", episode_length, global_step
                    )

        if (
            global_step > args.learning_starts
            and global_step % args.train_frequency == 0
        ):
            data = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                target_max = target_network(data.next_observations).max(dim=1).values
                td_target = data.rewards.flatten() + args.gamma * target_max * (
                    1 - data.dones.flatten()
                )

            old_value = q_network(data.observations).gather(
                dim=1, index=data.actions.long()
            ).squeeze()
            loss = nn.functional.mse_loss(td_target, old_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                steps_per_second = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_value.mean().item(), global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("charts/SPS", steps_per_second, global_step)

        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    model_path = Path("rom_cleanrl_dqn_cartpole.cleanrl_model")
    torch.save(
        {
            "env_id": "CartPole-v1",
            "model_state_dict": q_network.state_dict(),
        },
        model_path,
    )
    print(f"Training complete. Model saved to {model_path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    train(parse_args())