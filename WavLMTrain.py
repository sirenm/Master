import ast
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForSequenceClassification, Trainer, TrainingArguments
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
model = WavLMForSequenceClassification.from_pretrained("microsoft/wavlm-large", num_labels=6)
model.to(device)

class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        return {
            "input_values": audio_path,
            "labels": torch.tensor(label, dtype=torch.long)
        }

data_dir = "train.json"
label_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "SAD": 3,
    "HAP": 4,
    "NEU": 5
}
df = pd.read_json(data_dir)
df = df[df["emotion"].isin(label_map)]

labels = df["emotion"].map(label_map).tolist()

files = df["Features"].tolist()

dataset = EmotionDataset(files, labels)

eval_data_dir = "eval.json"
df = pd.read_json(eval_data_dir)
df = df[df["emotion"].isin(label_map)]

eval_labels = df["emotion"].map(label_map).tolist()

eval_files = df["Features"].tolist()

eval_dataset = EmotionDataset(eval_files, eval_labels)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset
)

trainer.train()
model.save_pretrained("TrainedModels/")
feature_extractor.save_pretrained("TrainedModels/")
