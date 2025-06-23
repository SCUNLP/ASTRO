import torch
from .RoBerta_model import RobertaScoreModel
from ..Agent_strategy_generating import Agent_strategy
from ..Agent_response_generating import Agent_assistant

class Eval_model():
    def __init__(self, device, api_key, url, prompt_data, response_model_name, model_load_pth, model_dir='./model_cache'):
        self.agent_strategy = Agent_strategy(api_key, url, prompt_data, "gpt-4o")
        self.agent_assistant = Agent_assistant(api_key, url, prompt_data, response_model_name)
        self.model = RobertaScoreModel(device)
        self.model.load_state_dict(torch.load(model_load_pth, map_location=device))
        self.strategy_set = []

    def set_strategy_set(self, env_strategy_data):
        self.strategy_set = self.agent_strategy.get_response(env_strategy_data)

    def get_response(self,chat_history,env_assistant_data):
        scores = []
        responses = []
        max_score_pos = 0
        for strategy_num in range(len(self.strategy_set)):
            strategy = self.strategy_set[strategy_num]
            response = self.agent_assistant.get_response(env_assistant_data,chat_history, strategy)
            score = self.model.get_response(chat_history, strategy, response)
            scores.append(score)
            responses.append(response)
            if scores[max_score_pos] < scores[strategy_num]:
                max_score_pos = strategy_num

        return self.strategy_set[max_score_pos],responses[max_score_pos]