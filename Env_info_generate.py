import json
from Agents import agent_api_base
from tqdm import tqdm
from utils import remove_json_tags


def generate_env_info(env_info_num, llm_model_name="gpt-3.5-turbo", data_pth="data/p4g_env_info.json",max_retries = 10):
    with open("api_key.json", "r") as f:
        api_key_data = json.load(f)

    api_key = api_key_data["api_key"]
    url = api_key_data["url"]

    agent = agent_api_base(api_key, url, llm_model_name)
    with open("Prompts/prompts.json","r") as f:
        prompt_data = json.load(f)
    prompt = prompt_data["env_info_generate"]

    try:
        with open(data_pth, "r") as f:
            data = json.load(f)
        data_len = len(data) + 1
    except FileNotFoundError:
        data = []
        data_len = 1

    for i in tqdm(range(data_len, data_len + env_info_num)):
        retries = 0
        while retries < max_retries:
            response = agent.base_agent.get_response_with_msg([], prompt)
            response = remove_json_tags(response)
            try:
                response_dict = json.loads(response)
                data.append(response_dict)
                break
            except json.JSONDecodeError:
                retries += 1
        else:
            raise ValueError(f"Max retries exceeded with invalid JSON response for index {i}")

    data_len += env_info_num

    with open(data_pth, "w+") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    generate_env_info(5)