import torch
import torchaudio
import torchaudio.transforms as T
from transformers import WavLMForCTC, Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, AutoProcessor
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForSequenceClassification, Trainer, TrainingArguments
import wandb
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
transcription_model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
transcription_processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("TrainedModels")
emotion_model = WavLMForSequenceClassification.from_pretrained("TrainedModels")
transcription_model.to(device)
emotion_model.to(device)

class EmotionModel():
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.correct_prediction = 0
        self.wrong_prediction = 0
        self.total_predictions = 0
        self.actual_predictions = []

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
           resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
           waveform = resampler(waveform)
        
        waveform = waveform.flatten().squeeze(0)

        transcription_inputs = transcription_processor(waveform, sampling_rate=16000, return_tensors="pt")
        transcription_inputs = {key: val.to(device) for key, val in transcription_inputs.items()}
        with torch.no_grad():
            logits = transcription_model(**transcription_inputs).logits
            print("Shape for waveform values: ", waveform.shape)
            inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
        transcription_predicted_ids = torch.argmax(logits, dim=-1)


        transcription = transcription_processor.batch_decode(transcription_predicted_ids)[0]

        with torch.no_grad():
            print("Shape for input values: ", input_values.shape)
            emotion_logits = emotion_model(input_values).logits
            predicted_emotion_class = torch.argmax(emotion_logits, dim=-1).item()

        if predicted_emotion_class == label:
            self.correct_prediction += 1
        else:
            self.wrong_prediction += 1

        self.actual_predictions.append(predicted_emotion_class)

        print("Transcription:", transcription)
        print("Predicted Emotion Class:", predicted_emotion_class)
        print("Actual Emotion Class:", label)
        self.total_predictions += 1


data_dir = "CremaTest"
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
emotionmodel = EmotionModel(files, labels)
print(len(files))
for index in range(len(files)):
    print(index)
    emotionmodel.__getitem__(index)

f1 = f1_score(emotionmodel.labels, emotionmodel.actual_predictions, average='macro')
print("Correct prediction:", emotionmodel.correct_prediction)
print("Wrong predictions:", emotionmodel.wrong_prediction)
print("Total predictions:", emotionmodel.total_predictions)
print("f1 score:", f1)
