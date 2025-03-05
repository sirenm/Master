import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForSequenceClassification, Trainer, TrainingArguments
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMForSequenceClassification.from_pretrained("microsoft/wavlm-base-plus", num_labels=6)
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
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        inputs = feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)

        return {
            "input_values": inputs["input_values"].flatten().squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

data_dir = "CremaTrain"
label_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "SAD": 3,
    "HAP": 4,
    "NEU": 5
}

labels = []
files = []

for filename in os.listdir(data_dir):
    if os.path.isfile(os.path.join(data_dir, filename)):
        full_path = os.path.join(data_dir, filename)
        file_name_list = filename.split('_')
        emotion = file_name_list[2]
        if emotion in label_map:
            labels.append(label_map[emotion])
            files.append(full_path)

dataset = EmotionDataset(files, labels)

eval_data_dir = "CremaEvaluation"

eval_labels = []
eval_files = []

for filename in os.listdir(eval_data_dir):
    if os.path.isfile(os.path.join(eval_data_dir, filename)):
        full_path = os.path.join(eval_data_dir, filename)
        file_name_list = filename.split('_')
        emotion = file_name_list[2]
        if emotion in label_map:
            eval_labels.append(label_map[emotion])
            eval_files.append(full_path)

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
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor
)

trainer.train()
model.save_pretrained("TrainedModels/")
feature_extractor.save_pretrained("TrainedModels/")
