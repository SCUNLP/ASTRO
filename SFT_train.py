import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datetime import datetime
from sklearn.model_selection import train_test_split
from Agents.RoBerta_model import RobertaScoreModel, ConversationDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class EntropyPenaltyLoss(nn.Module):
    def __init__(self, base_loss, alpha=0.1):
        super(EntropyPenaltyLoss, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha

    def forward(self, input, target):
        base_loss_value = self.base_loss(input, target)
        histogram = torch.histc(input, bins=10, min=input.min().item(), max=input.max().item())
        histogram = histogram / histogram.sum()
        entropy = -torch.sum(histogram * torch.log(histogram + 1e-9))
        return base_loss_value - self.alpha * entropy

def train(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0

    for history_input_ids, history_attention_mask, new_reply_input_ids, new_reply_attention_mask, target_score in dataloader:
        optimizer.zero_grad()
        predicted_score = model(history_input_ids, history_attention_mask, new_reply_input_ids,
                                new_reply_attention_mask).to(device)
        loss = criterion(predicted_score, target_score)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for history_input_ids, history_attention_mask, new_reply_input_ids, new_reply_attention_mask, target_score in dataloader:
            history_input_ids = history_input_ids.to(device)
            history_attention_mask = history_attention_mask.to(device)
            new_reply_input_ids = new_reply_input_ids.to(device)
            new_reply_attention_mask = new_reply_attention_mask.to(device)
            target_score = target_score.to(device)

            predicted_score = model(history_input_ids, history_attention_mask, new_reply_input_ids,
                                    new_reply_attention_mask)
            loss = criterion(predicted_score, target_score)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def SFT_train(chat_history_file_path, base_model_name, save_model_path, batch_size=50, num_epochs=150,
                        learning_rate=5e-6, weight_decay=0.1, early_stop=False, early_stopping_patience=20, alpha=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = RobertaScoreModel(model_name=base_model_name, device=device)
    model.to(device)

    # 加载数据集
    whole_dataset = ConversationDataset(chat_history_file_path, tokenizer=model.tokenizer, device=device)
    print(f"Size of dataset: {len(whole_dataset)}")
    train_size = int(0.8 * len(whole_dataset))
    val_size = len(whole_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(whole_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)



    # 使用自定义损失函数
    base_loss = nn.MSELoss()
    criterion = EntropyPenaltyLoss(base_loss, alpha=alpha)



    print(f'The model has {count_parameters(model):,} trainable parameters')

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, scheduler, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {current_time}')

        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), save_model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

    if not early_stop:
        torch.save(model.state_dict(), save_model_path)

    return model