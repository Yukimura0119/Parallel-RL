import argparse

import cv2
import torch
import wandb
from agent.agent import MultiStepDQNAgent
from env.atari import run_worker
from mpi4py import MPI

if __name__ == "__main__":
    # 限制 CPU 使用率
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    # MPI 初始化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results_parallel")
    parser.add_argument("--wandb-run-name", type=str, default="parallel_dqn_pong")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=8)  # 平行環境數量
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99998)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--target-update-frequency", type=int, default=2000)
    parser.add_argument("--replay-start-size", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=20000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=100000)
    args = parser.parse_args()

    if rank == 0:
        # Master Process
        print(f"Master process started. MPI Size: {size}")
        if size < args.num_envs + 1:
            print(
                f"Error: MPI size ({size}) is too small for {args.num_envs} environments + 1 master."
            )
        if size < args.num_envs + 1:
            print(
                f"Error: MPI size ({size}) is too small for {args.num_envs} environments + 1 master."
            )
            print(f"Please run with: mpirun -n {args.num_envs + 1} python main_mpi.py")
            print(
                f"Error: MPI size ({size}) is too small for {args.num_envs} environments + 1 master."
            )
            print(
                f"Error: MPI size ({size}) is too small for {args.num_envs} environments + 1 master."
            )

        wandb.init(project="DLP-Lab5-Parallel", name=args.wandb_run_name, config=args)

        agent = MultiStepDQNAgent(env_name="ALE/Pong-v5", args=args)
        try:
            agent.run(max_steps=args.max_steps)
        except KeyboardInterrupt:
            print("Stopping training...")
        finally:
            agent.envs.close()

    else:
        # Worker Process
        # Rank 1 對應 env_index 0, Rank 2 對應 env_index 1, ...
        # 確保只運行 num_envs 個 Worker
        if rank <= args.num_envs:
            run_worker(rank, "ALE/Pong-v5", args.seed)
        else:
            print(f"Rank {rank} is idle (extra process).")
