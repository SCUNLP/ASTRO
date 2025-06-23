from .Base_Api_Agent import agent_api_base
from utils import change_OpenAI_chat_history_to_str, remove_enclosing_symbols
import copy

class Agent_Rl_done(agent_api_base):
    def __init__(self, apikey, url, prompt_data, model_name="gpt-3.5-trubo") -> None:
        super().__init__(apikey, url, model_name)
        self.prompt=prompt_data["RL_done_check"]
        
        
    def get_response(self, chat_history, max_tries=15):
        chat_history = copy.copy(chat_history)
        history = []

        
        chat_history_str = change_OpenAI_chat_history_to_str(chat_history)
        input_msg = "["+chat_history_str+"]"+"\n"+self.prompt
        # history.append({"role": "user", "content": chat_history_str})
        history.append({"role": "user", "content": input_msg})

        valid_responses = ["reject", "negative reaction", "neutral", "positive reaction", "accept"]
        
        tries = 0
        while tries < max_tries:
            raw_response = self.base_agent.get_response_with_history(history)
            raw_response = remove_enclosing_symbols(raw_response)
            try:
                # 验证是否返回值在指定的五个词语中
                if raw_response not in valid_responses:
                    raise ValueError(f"Error: Response '{raw_response}' is not one of the valid responses")
                
                return raw_response
            except ValueError as e:
                tries += 1
                if tries >= max_tries:
                    raise ValueError(f"Error: It reaches the maximum number of attempts (Error output:{raw_response} \n Error history:{history})")

