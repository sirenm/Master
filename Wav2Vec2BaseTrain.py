import random
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import os
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=6
)
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
        inputs = feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding="max_length",
        truncation=True, max_length=39506)
       
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
for filename in sorted(os.listdir(data_dir)):
    if os.path.isfile(os.path.join(data_dir, filename)):
        full_path = os.path.join(data_dir, filename)
        emotion = filename.split('_')[2]
        if emotion in label_map:
            labels.append(label_map[emotion])
            files.append(full_path)

dataset = EmotionDataset(files, labels)

val_dir = "CremaValidation"
eval_labels = []
eval_files = []
for filename in sorted(os.listdir(val_dir)):
    if os.path.isfile(os.path.join(val_dir, filename)):
        full_path = os.path.join(val_dir, filename)
        emotion = filename.split('_')[2]
        if emotion in label_map:
            eval_labels.append(label_map[emotion])
            eval_files.append(full_path)

eval_dataset = EmotionDataset(eval_files, eval_labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    seed=42,
    dataloader_num_workers=0,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor 
)

trainer.train()
model.save_pretrained("Wav2Vec2Base/")
feature_extractor.save_pretrained("Wav2Vec2Base/")
