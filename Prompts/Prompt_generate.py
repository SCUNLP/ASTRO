import json
from .Base_prompts import base_prompt_data, meta_prompt_data, my_example
from Agents import agent_api_gpt
from utils import remove_enclosing_symbols


def generate_and_save_prompts(model_name, output_file="Prompts/prompts.json"):
    """
    Generates prompts using the provided parameters and saves them to a JSON file.

    Args:
        base_background (str): The base background information.
        apikey (str): The API key for the LLM.
        url (str): The URL for the LLM API.
        model_name (str): The name of the LLM model to use.
        output_file (str): The file path to save the prompts to. Defaults to "Prompts.json".
    """
    # Initialize example and prompts
    example = my_example
    prompts = {}
    prompts.update(base_prompt_data)

    with open("Background_infomation.txt", 'r', encoding='utf-8') as file:
        base_background = file.read()
    with open("api_key.json", "r") as f:
        api_key_data = json.load(f)

    api_key = api_key_data["api_key"]
    url = api_key_data["url"]

    

    # Format the env_info_generate prompt
    prompts["env_info_generate"] = prompts["env_info_generate"].format(
        base_background=base_background, example=example["env_info_example"]
    )

    # Initialize the LLM generator
    llm_generator = agent_api_gpt(api_key, url, model_name)

    prompts["resisting_strategies"] = llm_generator.get_response_with_msg(
        [], meta_prompt_data["resisting_strategies_generate"].format(
            base_background = base_background,
            resisting_strategies_output_form = example["resisting_strategies_output_form"] , 
            resisting_strategies_example = example["resisting_strategies_example"]
        )
    )
    print(meta_prompt_data["RL_done_check_generate"].format(base_background=base_background,my_example=example["RL_done_check_example"]))
    # Generate the RL_done_check prompt using the LLM
    prompts["RL_done_check"] = llm_generator.get_response_with_msg(
        [], meta_prompt_data["RL_done_check_generate"].format(base_background=base_background,
                                                              my_example=example["RL_done_check_example"])
    )

    prompts["RL_done_check"] = remove_enclosing_symbols(prompts["RL_done_check"])

    # Save the prompts to a JSON file
    with open(output_file, "w+") as f:
        json.dump(prompts, f,indent=4)
