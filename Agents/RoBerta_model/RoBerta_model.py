import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Prompts import Prompt_generate
import copy
import json
from utils import change_OpenAI_form_to_str,change_OpenAI_chat_history_to_str


class TransformerHead(nn.Module):
    def __init__(self, hidden_size, num_layers=2, num_heads=8, dropout=0.3):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        
        self.fnn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.GELU(),
            
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.Dropout(dropout)
        )
        
        self.linear = nn.Linear(hidden_size//4, 1)
        self.output_f = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fnn(x)
        return self.linear(x) * 5

class RobertaScoreModel(nn.Module):
    def __init__(self, device, model_name='roberta-base',model_dir='./model_cache'):
        super(RobertaScoreModel, self).__init__()
        with open("Prompts/prompts.json","r") as f:
            prompt_data = json.load(f)
        self.prompt1 = prompt_data["score_get1"]
        self.prompt2 = prompt_data["score_get2"]
        self.device = device
        self.model_dir = model_dir

        self.model1 = AutoModelForMaskedLM.from_pretrained(model_name,cache_dir=self.model_dir,output_hidden_states=True)
        self.model2 = AutoModelForMaskedLM.from_pretrained(model_name,cache_dir=self.model_dir,output_hidden_states=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=self.model_dir)
        self.transformer_head = TransformerHead(hidden_size=self.model1.config.hidden_size * 2)


    def forward(self, history_input_ids,history_attention_mask, new_reply_input_ids,new_reply_attention_mask):
        history_outputs = self.model1(history_input_ids, attention_mask=history_attention_mask)
        new_reply_outputs = self.model2(new_reply_input_ids, attention_mask=new_reply_attention_mask)
        

        history_hidden_states = history_outputs.hidden_states[-1]
        new_reply_hidden_states = new_reply_outputs.hidden_states[-1]


        combined_hidden_states = torch.cat((history_hidden_states[:, 0, :], new_reply_hidden_states[:, 0, :]), dim=1)
        combined_hidden_states = combined_hidden_states.unsqueeze(1)

        score = self.transformer_head(combined_hidden_states)
        
        return score

    def get_response(self, history, strategy, msg):
        # Convert history and msg to the format required by the model
        history_text,msg_text = change_OpenAI_form_to_str(history,{"role": "assistant", "content": msg, "strategy": strategy})

        history_tokens = self.tokenizer(history_text, return_tensors='pt', padding=True, truncation=True)
        msg_tokens = self.tokenizer(msg_text, return_tensors='pt', padding=True, truncation=True)

        history_input_ids = history_tokens['input_ids'].to(self.device)
        history_attention_mask = history_tokens['attention_mask'].to(self.device)
        msg_input_ids = msg_tokens['input_ids'].to(self.device)
        msg_attention_mask = msg_tokens['attention_mask'].to(self.device)

        score = self.forward(history_input_ids, history_attention_mask, msg_input_ids, msg_attention_mask)
        return score