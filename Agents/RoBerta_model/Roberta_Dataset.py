import json
import torch
import os
from torch.utils.data import Dataset

from utils import change_OpenAI_form_to_bert_dataset,transform_score_value


class ConversationDataset(Dataset):
    def __init__(self, folder_path, tokenizer,device, max_length=512):
        self.chat_histories = []

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device

        json_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    json_files.append(file_path)

        #print(json_files)

        for json_file in json_files:
            with open(json_file,"r") as f:
                json_data = json.load(f)
                json_data = change_OpenAI_form_to_bert_dataset(json_data)
                self.chat_histories=self.chat_histories + json_data

    def __len__(self):
        return len(self.chat_histories)

    def __getitem__(self, idx):
        chat_history = self.chat_histories[idx]["chat_history"]
        response = self.chat_histories[idx]["latest_response"]
        score = self.chat_histories[idx]["score"]
        score = transform_score_value(float(score))

        chat_history = self.tokenizer(chat_history,add_special_tokens=True, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        response = self.tokenizer(response,add_special_tokens=True, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        

        return (
            chat_history['input_ids'].squeeze(0).to(self.device),
            chat_history['attention_mask'].squeeze(0).to(self.device),
            response['input_ids'].squeeze(0).to(self.device),
            response['attention_mask'].squeeze(0).to(self.device),
            torch.tensor(score, dtype=torch.float).to(self.device)
        )
