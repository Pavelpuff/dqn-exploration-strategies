import argparse

import gymnasium as gym
import torch

import wandb
from src.models import agent as agents
from src.utils.seed import set_seed


def main(args: argparse.Namespace) -> None:
    set_seed(42)

    wandb.init(project="dqn-exploration", config=vars(args))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Using hardware accelerator: {device}")

    # Infer state and action dimensions dynamically from env
    temp_env = gym.make(args.gym_id)
    args.state_dim = temp_env.observation_space.shape[0]
    args.action_dim = temp_env.action_space.n
    temp_env.close()

    # Strategy Selection Mapping
    strategy_map = {
        "eps_greedy": agents.EpsilonGreedyDQN,
        "eps_annealing": agents.EpsilonGreedyWithAnnealingDQN,
        "boltzmann": agents.BoltzmannDQN,
        "max_boltzmann": agents.MaxBoltzmannDQN,
    }

    if args.strategy not in strategy_map:
        raise ValueError(f"Unknown strategy {args.strategy}")

    print(f"[*] Initializing {args.strategy} DQN Agent...")
    agent_class = strategy_map[args.strategy]
    agent = agent_class(args, device)

    print("[*] Starting training loop...")
    agent.train()

    wandb.finish()
    print("[*] Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent on LunarLander")
    parser.add_argument(
        "--strategy",
        type=str,
        default="eps_annealing",
        choices=["eps_greedy", "eps_annealing", "boltzmann", "max_boltzmann"],
    )
    parser.add_argument("--gym_id", type=str, default="LunarLander-v3")
    parser.add_argument("--n_seeds", type=int, default=6)
    parser.add_argument("--n_evaluate_episodes", type=int, default=5)
    parser.add_argument("--n_training_steps", type=int, default=100000)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--init_steps", type=int, default=10000)
    parser.add_argument("--target_update_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=7e-4)

    # Exploration hyperparameters
    parser.add_argument("--start_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=0.1)
    parser.add_argument("--anneal_steps", type=int, default=30000)
    parser.add_argument("--start_temperature", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
