import torch
import torchaudio
from transformers import WavLMForCTC, Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, AutoProcessor
import os
import numpy as np
from sklearn.metrics import f1_score
import random
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

transcription_model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
transcription_processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("WavLMBase15")
emotion_model = WavLMForSequenceClassification.from_pretrained("WavLMBase15")
transcription_model.to(device)
emotion_model.to(device)

class EmotionModel:
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.correct_prediction = 0
        self.wrong_prediction = 0
        self.total_predictions = 0
        self.actual_predictions = []

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        waveform = waveform.flatten().squeeze(0)

        with torch.no_grad():
            transcription_inputs = transcription_processor(waveform, sampling_rate=16000, return_tensors="pt")
            transcription_inputs = {k: v.to(device) for k, v in transcription_inputs.items()}
            logits = transcription_model(**transcription_inputs).logits
            transcription_ids = torch.argmax(logits, dim=-1)
            transcription = transcription_processor.batch_decode(transcription_ids)[0]

            inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=16000)
            input_values = inputs["input_values"].to(device)
            emotion_logits = emotion_model(input_values).logits
            predicted_emotion_class = torch.argmax(emotion_logits, dim=-1).item()

        if predicted_emotion_class == label:
            self.correct_prediction += 1
        else:
            self.wrong_prediction += 1

        self.actual_predictions.append(predicted_emotion_class)
        self.total_predictions += 1

        print("Transcription:", transcription)
        print("Predicted Emotion Class:", predicted_emotion_class)
        print("Actual Emotion Class:", label)

data_dir = "CremaTest"
label_map = {"ANG": 0, "DIS": 1, "FEA": 2, "SAD": 3, "HAP": 4, "NEU": 5}
labels, files = [], []

for filename in sorted(os.listdir(data_dir)):
    if os.path.isfile(os.path.join(data_dir, filename)):
        full_path = os.path.join(data_dir, filename)
        emotion = filename.split('_')[2]
        if emotion in label_map:
            labels.append(label_map[emotion])
            files.append(full_path)

emotionmodel = EmotionModel(files, labels)

for index in range(len(files)):
    emotionmodel.__getitem__(index)

f1 = f1_score(emotionmodel.labels, emotionmodel.actual_predictions, average='macro')
print("Correct prediction:", emotionmodel.correct_prediction)
print("Wrong predictions:", emotionmodel.wrong_prediction)
print("Total predictions:", emotionmodel.total_predictions)
print("f1 score:", f1)
print("Accurancy:", emotionmodel.correct_prediction/emotionmodel.total_predictions)

results_data = {
    "Actual Labels": emotionmodel.labels,
    "Predicted Labels": emotionmodel.actual_predictions
}

df_results = pd.DataFrame(results_data)

df_results.to_csv("WavLMBase15.csv", index=False)
