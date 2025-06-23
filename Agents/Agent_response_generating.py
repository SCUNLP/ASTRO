from .Base_Api_Agent import agent_api_base
import copy

class Agent_assistant(agent_api_base):
    def __init__(self, apikey, url, prompt_data, model_name="gpt-3.5-trubo") -> None:
        super().__init__(apikey, url, model_name)
        self.prompt=prompt_data["response_generating"]
        # if "assiassisitant_bg_info" in prompt_data:
        #     self.env_data = prompt_data["assisitant_bg_info"]
        # else:
        #     self.env_data = prompt_data["env_info"]
        
        
    def get_response(self, env_data, history, strategy, max_tries=3):
        history=copy.copy(history)
        system_msg=self.prompt
        system_msg=system_msg.format(env_info=env_data,strategy=strategy)
        # print(system_msg)
        history.append({"role":"user","content":system_msg})

        # print(f"assistant: \n{system_msg}")
        # tries = 0
        response = self.base_agent.get_response_with_history(history)
        return response

        # raise ValueError("Error: It reaches the maximum number of attempts (In response generating)")
