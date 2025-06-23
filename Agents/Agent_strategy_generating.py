from .Base_Api_Agent import agent_api_base
from utils import remove_json_tags
import json

class Agent_strategy(agent_api_base):
    def __init__(self, apikey, url, prompt_data, model_name = "gpt-3.5-trubo") -> None:
        super().__init__(apikey,url,model_name)
        self.prompt=prompt_data["strategy_generating"]
    
    def get_response(self, env_tuple, max_tries=10):
        env_info, user_info = env_tuple
        msg=self.prompt
        msg=msg.format(env_info=env_info,user_info=user_info)
        tries = 0
        while tries < max_tries:
            raw_response = self.base_agent.get_response_with_msg([],msg)
            raw_response = remove_json_tags(raw_response)
            try:
                response_dict = json.loads(raw_response)
                return response_dict
            except json.JSONDecodeError:
                tries += 1

        raise ValueError("Error: It reaches the maximum number of attempts (In strategy generating)")
