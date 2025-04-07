import torch
import torchaudio
from transformers import WavLMForCTC, Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, AutoProcessor
import os
import numpy as np
from sklearn.metrics import f1_score
import random
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("WavLMLarge152ndtry")
emotion_model = WavLMForSequenceClassification.from_pretrained("WavLMLarge152ndtry")
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
        self.happy_to_angry = []
        self.happy_to_fear = []
        self.happy_to_sad = []
        self.happy_to_disgust = []

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

            inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
            emotion_logits = emotion_model(input_values).logits
            predicted_emotion_class = torch.argmax(emotion_logits, dim=-1).item()

        if predicted_emotion_class == label:
            self.correct_prediction += 1
        else:
            self.wrong_prediction += 1
        
        if label == 4 and predicted_emotion_class == 0:
            self.happy_to_angry.append(audio_path)
        
        if label == 4 and predicted_emotion_class == 2:
            self.happy_to_fear.append(audio_path)
        
        if label == 4 and predicted_emotion_class == 1:
            self.happy_to_disgust.append(audio_path)
        
        if label == 4 and predicted_emotion_class == 3:
            self.happy_to_sad.append(audio_path)

        self.actual_predictions.append(predicted_emotion_class)
        self.total_predictions += 1

        """ print("Transcription:", transcription)
        print("Predicted Emotion Class:", predicted_emotion_class)
        print("Actual Emotion Class:", label) """

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
print("Accurancy:", emotionmodel.correct_prediction/emotionmodel.total_predictions)
print("f1 score:", f1)

GENDERMAP = {
    "1001": "Male",
    "1002": "Female",
    "1003": "Female",
    "1004": "Female",
    "1005": "Male",
    "1006": "Female",
    "1007": "Female",
    "1008": "Female",
    "1009": "Female",
    "1010": "Female",
    "1011": "Male",
    "1012": "Female",
    "1013": "Female",
    "1014": "Male",
    "1015": "Male",
    "1016": "Male",
    "1017": "Male",
    "1018": "Female",
    "1019": "Male",
    "1020": "Female",
    "1021": "Female",
    "1022": "Male",
    "1023": "Male",
    "1024": "Female",
    "1025": "Female",
    "1026": "Male",
    "1027": "Male",
    "1028": "Female",
    "1029": "Female",
    "1030": "Female",
    "1031": "Male",
    "1032": "Male",
    "1033": "Male",
    "1034": "Male",
    "1035": "Male",
    "1036": "Male",
    "1037": "Female",
    "1038": "Male",
    "1039": "Male",
    "1040": "Male",
    "1041": "Male",
    "1042": "Male",
    "1043": "Female",
    "1044": "Male",
    "1045": "Male",
    "1046": "Female",
    "1047": "Female",
    "1048": "Male",
    "1049": "Female",
    "1050": "Male",
    "1051": "Male",
    "1052": "Female",
    "1053": "Female",
    "1054": "Female",
    "1055": "Female",
    "1056": "Female",
    "1057": "Male",
    "1058": "Female",
    "1059": "Male",
    "1060": "Female",
    "1061": "Female",
    "1062": "Male",
    "1063": "Female",
    "1064": "Male",
    "1065": "Male",
    "1066": "Male",
    "1067": "Male",
    "1068": "Male",
    "1069": "Male",
    "1070": "Male",
    "1071": "Male",
    "1072": "Female",
    "1073": "Female",
    "1074": "Female",
    "1075": "Female",
    "1076": "Female",
    "1077": "Male",
    "1078": "Female",
    "1079": "Female",
    "1080": "Male",
    "1081": "Male",
    "1082": "Female",
    "1083": "Male",
    "1084": "Female",
    "1085": "Male",
    "1086": "Male",
    "1087": "Male",
    "1088": "Male",
    "1089": "Female",
    "1090": "Male",
    "1091": "Female"
}

print("Happy => Angry:", emotionmodel.happy_to_angry)
male_count = 0
total_count = len(emotionmodel.happy_to_angry)
intensity_count = 0
for filepath in emotionmodel.happy_to_angry:
    actor_id = filepath.split('/')[1].split('_')[0]  # Extract '1007' from 'CremaTest/1007_TIE_HAP_XX.wav'
    if GENDERMAP.get(actor_id) == "Male":
        male_count += 1
    intensity = filepath.split('/')[1].split('_')[3]
    if intensity != "XX.wav":
        intensity_count += 1

print(f"{male_count}/{total_count}")
print(f"Intensity: {intensity_count}/{total_count}")

print("Happy => Fear:", emotionmodel.happy_to_fear)

male_count = 0
intensity_count = 0
total_count = len(emotionmodel.happy_to_fear)

for filepath in emotionmodel.happy_to_fear:
    actor_id = filepath.split('/')[1].split('_')[0]  # Extract '1007' from 'CremaTest/1007_TIE_HAP_XX.wav'
    intensity = filepath.split('/')[1].split('_')[3]
    if intensity != "XX.wav":
        intensity_count += 1
    if GENDERMAP.get(actor_id) == "Male":
        male_count += 1

print(f"{male_count}/{total_count}")
print(f"Intensity: {intensity_count}/{total_count}")

print("Happy => Disgust:", emotionmodel.happy_to_disgust)

male_count = 0
total_count = len(emotionmodel.happy_to_disgust)
intensity_count = 0
for filepath in emotionmodel.happy_to_disgust:
    actor_id = filepath.split('/')[1].split('_')[0]  # Extract '1007' from 'CremaTest/1007_TIE_HAP_XX.wav'
    intensity = filepath.split('/')[1].split('_')[3]
    if intensity != "XX.wav":
        intensity_count += 1
    if GENDERMAP.get(actor_id) == "Male":
        male_count += 1

print(f"{male_count}/{total_count}")
print(f"Intensity: {intensity_count}/{total_count}")

results_data = {
    "Actual Labels": emotionmodel.labels,
    "Predicted Labels": emotionmodel.actual_predictions
}

df_results = pd.DataFrame(results_data)

df_results.to_csv("WavLMLarge15.csv", index=False)