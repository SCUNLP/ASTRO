import json
from Agents import Agent_assistant, Agent_strategy, Agent_user, Agent_score

def generate_SFT_chat_history(info_num, max_chat_rounds, base_model_name, save_file_path):
    with open("Prompts/prompts.json") as f:
        prompt_data = json.load(f)

    with open("api_key.json", "r") as f:
        api_key_data = json.load(f)

    api_key = api_key_data["api_key"]
    url = api_key_data["url"]
    info_num=str(info_num)

    with open("data/env_info.json", "r") as f:
        env_data = json.load(f)
    env_data_strategy = (env_data[info_num]["env_info"], env_data[info_num]["user_info"])
    env_data_user = (env_data[info_num]["user_bg_info"], env_data[info_num]["user_info2"])
    env_data_assistant =env_data[info_num]["assisitant_bg_info"]

    # print("Init model...")

    agent_strategy = Agent_strategy(api_key, url, prompt_data, base_model_name)
    agent_assistant = Agent_assistant(api_key, url, prompt_data, base_model_name)
    agent_user = Agent_user(api_key, url, prompt_data, base_model_name)
    agent_score = Agent_score(api_key, url, prompt_data, base_model_name)

    chat_data = {}


    chat_history = []
    full_history = []
    strategy_set = agent_strategy.get_response(env_data_strategy)
    chat_data["strategy"] = strategy_set
    chat_data["score"] = []
    chat_data["full_history"] = []


    for chat_round in range(max_chat_rounds):
        user_response = agent_user.get_response(env_data_user, chat_history)
        chat_history.append({"role": "user", "content": user_response})
        full_history.append({"role": "user", "content": user_response})
        score_list = []

        max_pos = 0
        pos = 0
        for strategy in strategy_set:
            assistant_response = agent_assistant.get_response(env_data_assistant, chat_history, strategy)
            raw_score = agent_score.get_response(chat_history, strategy, assistant_response)
            score_list.append((assistant_response, raw_score))
            if score_list[pos][1] > score_list[max_pos][1]:
                max_pos = pos
            pos += 1
        chat_data["score"].append(score_list)
        chat_history.append({"role": "assistant", "content": score_list[max_pos][0]})
        full_history.append({"role": "assistant", "content": score_list[max_pos][0], "strategy": strategy_set[max_pos], "score": score_list[max_pos][1]})

    chat_data["history"] = chat_history
    chat_data["full_history"] = full_history

    with open(save_file_path, "w") as json_file:
        json.dump(chat_data, json_file,indent=4)