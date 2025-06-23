from .Base_Api_Agent import agent_api_base
import json
import copy
from utils import score_get

class Agent_score(agent_api_base):
    def __init__(self, apikey, url, prompt_data, model_name="gpt-3.5-trubo") -> None:
        super().__init__(apikey, url, model_name)
        self.prompt1 = prompt_data["score_get1"]
        self.prompt2 = prompt_data["score_get2"]
        
    def get_response(self, history, strategy, msg, max_tries=10):
        history = copy.copy(history)
        history.append({"role":"assistant","content":msg})
        system_msg=self.prompt1
        system_msg=system_msg.format(strategy_now=strategy)
        system_msg=system_msg+self.prompt2
        history.append({"role":"user","content":system_msg})

        tries = 0

        while tries < max_tries:
            raw_response = self.base_agent.get_response_with_history(history)
            try:
                
                response_dict = json.loads(raw_response)
                for key, value in response_dict.items():
                    try:
                        response_dict[key] = float(value)
                    except ValueError:
                        raise ValueError("Error: Value cannot be converted to float")
                return score_get(response_dict)
            except (json.JSONDecodeError, ValueError) as e:
                tries += 1
                if tries >= max_tries:
                    raise ValueError("Error: It reaches the maximum number of attempts (In score get)")
