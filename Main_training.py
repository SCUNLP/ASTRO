import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


import torch
import argparse
from datetime import datetime
from SFT_train import SFT_train
from RL_train import RL_train


def SFT_train_seperate(args, sft_save_model_path):
    # SFT 训练参数
    sft_params = {
        'chat_history_file_path': args.chat_history_file_path,
        'base_model_name': args.base_model_name,
        'save_model_path': sft_save_model_path,
        'batch_size': args.batch_size,
        'num_epochs': args.sft_num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stop': args.early_stop,
        'early_stopping_patience': args.early_stopping_patience,
        'alpha': args.alpha
    }

    # 开始训练
    print("Starting SFT training...")
    start_time = datetime.now()
    SFT_train(**sft_params)
    end_time = datetime.now()
    print(f"SFT training completed in {end_time - start_time}")

    

def RL_train_seperate(args,base_dir,sft_save_model_path):
    # RL模型保存路径
    rl_save_model_dir = os.path.join(base_dir, "rl_trained_model")
    os.makedirs(rl_save_model_dir, exist_ok=True)

    # RL训练日志保存路径
    rl_log_save_dir = os.path.join(base_dir, "rl_training_log")
    os.makedirs(rl_log_save_dir, exist_ok=True)

    # RL 训练参数
    rl_params = {
        'llm_model_name': args.llm_model_name,
        'model_load_pth': sft_save_model_path,
        'model_save_pth': rl_save_model_dir,
        'log_save_pth': rl_log_save_dir,
        'LR': args.LR,
        'num_episodes': args.rl_num_episodes,
        'gamma': args.gamma
    }


    print("Starting RL training...")
    start_time = datetime.now()
    RL_train(**rl_params)
    end_time = datetime.now()
    print(f"RL training completed in {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for SFT and RL models")

    # 添加任务名称参数
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task')

    # 添加其他必要的参数
    parser.add_argument('--chat_history_file_path', type=str, default="data/SFT-chat_history/", help='Path to chat history file')
    parser.add_argument('--base_model_name', type=str, default="roberta-base", help='Base model name')

    # SFT训练参数
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for SFT training')
    parser.add_argument('--sft_num_epochs', type=int, default=50, help='Number of epochs for SFT training')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate for SFT training')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for SFT training')
    parser.add_argument('--early_stop', type=bool, default=True, help='Enable early stopping for SFT training')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha parameter for SFT training')

    # RL训练参数
    parser.add_argument('--llm_model_name', type=str, default="gpt-4o", help='LLM model name for environment')
    parser.add_argument('--LR', type=float, default=0.01, help='Learning rate for RL training')
    parser.add_argument('--rl_num_episodes', type=int, default=100, help='Number of episodes for RL training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma parameter for RL training')

    args = parser.parse_args()

    # 设置目录结构
    base_dir = os.path.join("model", args.task_name)
    os.makedirs(base_dir, exist_ok=True)

    # SFT模型保存路径
    sft_save_model_path = os.path.join(base_dir, "SFT_trained_model.pth")
    # SFT_train_seperate(args, sft_save_model_path)
    # sft_save_model_path = os.path.join(base_dir, "rl_trained_model.pth")
    model , score_list = RL_train_seperate(args,base_dir,sft_save_model_path)
