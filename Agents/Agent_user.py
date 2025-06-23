from .Base_Api_Agent import agent_api_base
import copy

class Agent_user(agent_api_base):
    def __init__(self, apikey, url, prompt_data, model_name = "gpt-3.5-trubo") -> None:
        super().__init__(apikey,url,model_name)
        self.resisting_strategies = prompt_data["resisting_strategies"]
        self.prompt = prompt_data["user"]

    def get_response(self, env_data, history):
        history = copy.copy(history)
        new_history = []
        

        for msg in history:
            if msg["role"] == "user":
                new_history.append({"role":"assistant","content":msg["content"]})
            elif msg["role"] == "assistant":
                new_history.append({"role":"user","content":msg["content"]})

        env_info, user_info = env_data

        system_msg = self.prompt
        # system_msg=system_msg.format(env_info=env_info,user_info=user_info,resisting_strategies_str = self.resisting_strategies)
        system_msg=system_msg.format(env_info=env_info,user_info=user_info)

        new_history.append({"role":"user","content":system_msg})
        # print(f"user : \n{new_history}")
        response = self.base_agent.get_response_with_history(new_history)

        return response