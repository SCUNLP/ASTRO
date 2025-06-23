import argparse
import os
import random
import json
from datetime import datetime
from Env_info_generate import generate_env_info
from SFTData_generate import generate_SFT_chat_history
from Prompts import generate_and_save_prompts
from tqdm import tqdm

def get_next_file_number(directory):
    """Get the next file number in the specified directory."""
    existing_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not existing_files:
        return 1
    existing_numbers = [int(os.path.splitext(f)[0]) for f in existing_files]
    return max(existing_numbers) + 1
def main():
    parser = argparse.ArgumentParser(description="Data preparing script")

    parser.add_argument('--env_info_num',type=int, default=10, help="")#15
    parser.add_argument('--llm_model_name', type=str, default="gpt-4o-2024-08-06", help='LLM model name for environment')

    parser.add_argument("--num_files", type=int, default=50, help="Number of chat dialogues to generate.")#50
    parser.add_argument("--max_chat_rounds", type=int, default=8, help="Maximum number of chat rounds per file.")

    args = parser.parse_args()

    print("Generate prompts...")
    generate_and_save_prompts(args.llm_model_name)

    return
    print("Generate environment information...")
    generate_env_info(args.env_info_num,args.llm_model_name)

    output_dir = "data/SFT-chat_history"

    return

    os.makedirs(output_dir, exist_ok=True)

    with open("data/env_info.json", "r") as f:
        env_data = json.load(f)

    print("Generate SFT chat history data...")
    for _ in tqdm(range(args.num_files)):
        info_num = random.randint(1, len(env_data))
        file_number = get_next_file_number(output_dir)
        save_file_path = os.path.join(output_dir, f"{file_number}.json")
        generate_SFT_chat_history(info_num, args.max_chat_rounds, args.llm_model_name, save_file_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Generated SFT chat history in file: {save_file_path}")


if __name__ == "__main__":
    main()