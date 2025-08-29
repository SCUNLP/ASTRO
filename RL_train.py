import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from Agents.RoBerta_model import RobertaScoreModel, ConversationDataset
from RL import ConversationEnv
import json
from datetime import datetime
import os
import argparse
from transformers import AutoTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def RL_train(llm_model_name, model_load_pth,  model_save_pth, log_save_pth, LR=0.01, num_episodes=50, gamma=0.99):
    score_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaScoreModel(device)
    model.load_state_dict(torch.load(model_load_pth, map_location=device))
    model.to(device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained('roberta-base',cache_dir='./model_cache')
    
    start_info_num = 1
    env = ConversationEnv(model, tokenizer, device, start_info_num, llm_model_name)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode = 0
    while episode < num_episodes:
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            responses = []
            detached_scores = []

            with torch.no_grad():
                for action_strategy_num in range(len(env.strategy)):
                    action_strategy = env.strategy[action_strategy_num]
                    response = env.agent_assistant.get_response(env.env_data_assistant, env.chat_history, action_strategy)
                    score = model.get_response(env.chat_history, action_strategy, response)
                    detached_scores.append(score.detach())
                    responses.append(response)

            scores_tensor = torch.cat(detached_scores).squeeze(1).to(device)

            probs = torch.softmax(scores_tensor, dim=-1)
            m = Categorical(probs)
            action = m.sample()

            selected_strategy = env.strategy[action.item()]
            selected_response = responses[action.item()]
            selected_score = model.get_response(env.chat_history, selected_strategy, selected_response)
            selected_score = selected_score.squeeze()


            logits = torch.zeros_like(scores_tensor)
            logits[action.item()] = selected_score

            probs_for_backward = torch.softmax(logits, dim=-1)
            m = Categorical(probs_for_backward)
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)


            chosen_response = responses[action.item()]
            state, reward, done, _ = env.step((action.item(), chosen_response))
            rewards.append(reward)

        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = 0
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss -= log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        reward_avg = sum(rewards)/len(rewards)
        if reward_avg == 0.0:
            episode += 1
            continue
        score_list.append(reward_avg)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Episode {episode+1}\tTime: {current_time}\tReward Avg: {reward_avg}')

        if (episode + 1) % 5 == 0:
            if not os.path.exists(log_save_pth):
                os.makedirs(log_save_pth)
            chat_history_path = os.path.join(log_save_pth, f"full_chat_history_epoch_{episode + 1}.json")
            with open(chat_history_path, "w") as f:
                json.dump(env.full_history, f, ensure_ascii=False, indent=4)

        if (episode + 1) % 10 == 0:
            save_model_pth_in_process = os.path.join(model_save_pth, f"model_epoch_{episode + 1}.pth")
            torch.save(model.state_dict(), save_model_pth_in_process)

        episode += 1

        torch.cuda.empty_cache()

    save_model_pth_final = os.path.join(model_save_pth, "rl_trained_model.pth")
    torch.save(model.state_dict(), save_model_pth_final)

    return model, score_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for RL")

    parser.add_argument('--task_name', type=str, required=True, help='Name of the task')

    parser.add_argument('--llm_model_name', type=str, default="gpt-4o", help='LLM model name for environment')
    parser.add_argument('--LR', type=float, default=0.01, help='Learning rate for RL training')
    parser.add_argument('--rl_num_episodes', type=int, default=100, help='Number of episodes for RL training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma parameter for RL training')
    parser.add_argument('--load_model_dir' , type=str, default="model/test3/rl_trained_model/rl_trained_model.pth")

    args = parser.parse_args()

    base_dir = os.path.join("model", args.task_name)
    os.makedirs(base_dir, exist_ok=True)

    rl_save_model_dir = os.path.join(base_dir, "rl_trained_model")
    os.makedirs(rl_save_model_dir, exist_ok=True)

    rl_log_save_dir = os.path.join(base_dir, "rl_training_log")
    os.makedirs(rl_log_save_dir, exist_ok=True)

    rl_params = {
        'llm_model_name': args.llm_model_name,
        'model_load_pth': args.load_model_dir,
        'model_save_pth': rl_save_model_dir,
        'log_save_pth': rl_log_save_dir,
        'LR': args.LR,
        'num_episodes': args.rl_num_episodes,
        'gamma': args.gamma
    }


    print("Starting RL training...")
    start_time = datetime.now()
    model , score_list = RL_train(**rl_params)
    end_time = datetime.now()
    print(f"RL training completed in {end_time - start_time}")