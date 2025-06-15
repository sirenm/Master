import torch
import torchaudio
from transformers import WavLMForCTC, Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, AutoProcessor
import os
import numpy as np
from sklearn.metrics import f1_score
import random
import pandas as pd
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

label_map = {"ANG": 0, "DIS": 1, "FEA": 2, "SAD": 3, "HAP": 4, "NEU": 5}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transcription_model = WavLMForCTC.from_pretrained(
    "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
)
transcription_processor = AutoProcessor.from_pretrained(
    "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
)

model_path = "WavLMLarge"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
emotion_model = WavLMForSequenceClassification.from_pretrained(model_path)

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
        self.happyfilepaths = []
        self.happypredictedlabel = []
        self.predicted_probabilities = []

    def evaluate_and_store_predictions(self):
        results = []

        for i, audio_path in enumerate(self.file_paths):
            label = self.labels[i]
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )(waveform)
            waveform = waveform.flatten().squeeze(0)

            with torch.no_grad():
                # Transcription
                transcription_inputs = transcription_processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                transcription_inputs = {k: v.to(device) for k, v in transcription_inputs.items()}
                logits = transcription_model(**transcription_inputs).logits
                transcription_ids = torch.argmax(logits, dim=-1)
                transcription = transcription_processor.batch_decode(transcription_ids)[0]

                # Emotion classification
                inputs = feature_extractor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                input_values = inputs["input_values"].to(device)
                emotion_logits = emotion_model(input_values).logits
                probs = F.softmax(emotion_logits, dim=1)
                predicted_classes = torch.argmax(emotion_logits, dim=1)

                # Update counts
                if predicted_classes.item() == label:
                    self.correct_prediction += 1
                else:
                    self.wrong_prediction += 1

                # If true "happy", store detailed info
                if label == 4:
                    self.predicted_probabilities.append(probs.squeeze().cpu().numpy())
                    self.happyfilepaths.append(audio_path)
                    self.happypredictedlabel.append(predicted_classes.item())

                # Map happy misclassifications
                if label == 4 and predicted_classes.item() == 0:
                    self.happy_to_angry.append(audio_path)
                if label == 4 and predicted_classes.item() == 2:
                    self.happy_to_fear.append(audio_path)
                if label == 4 and predicted_classes.item() == 1:
                    self.happy_to_disgust.append(audio_path)
                if label == 4 and predicted_classes.item() == 3:
                    self.happy_to_sad.append(audio_path)

                # Build per-sample results
                logits_cpu = emotion_logits.cpu()
                max_confidences, _ = torch.max(probs, dim=1)
                for idx_sample in range(logits_cpu.size(0)):
                    full_logits = logits_cpu[idx_sample].tolist()
                    rounded_logits = [round(x, 2) for x in full_logits]
                    second_idx = int(np.argsort(rounded_logits)[-2])
                    confidence = round(float(max_confidences[idx_sample].cpu()), 2)

                    results.append({
                        "true_label": label,
                        "logits": rounded_logits,
                        "predicted_label": int(predicted_classes[idx_sample]),
                        "confidence": confidence,
                        "second_highest_index": second_idx
                    })
                self.actual_predictions.append(predicted_classes.item())
                self.total_predictions += 1

        # Post-process results
        inv_label_map = {v: k for k, v in label_map.items()}  
        for res in results:
            res["true_label_name"] = inv_label_map[res["true_label"]]
            res["predicted_label_name"] = inv_label_map[res["predicted_label"]]
            res["second_highest_label_name"] = inv_label_map[res["second_highest_index"]]
            c = res["confidence"]
            if c >= 0.9:
                res["certainty"] = "Confident"
            elif c >= 0.5:
                res["certainty"] = "Uncertain"
            else:
                res["certainty"] = "Very Uncertain"

        df_results = pd.DataFrame(results)
        df_display = df_results[[
            "true_label_name", "logits", "predicted_label_name",
            "second_highest_index", "second_highest_label_name",
            "confidence", "certainty"
        ]]

        output_csv = model_path + "_predictions_with_logits.csv"
        df_display.to_csv(output_csv, index=False)
        print(f"Predictions with logits saved to {output_csv}")

    def __getitem__(self, idx):
        # Same logic as evaluate_and_store_predictions for a single item
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)
        waveform = waveform.flatten().squeeze(0)

        with torch.no_grad():
            transcription_inputs = transcription_processor(
                waveform, sampling_rate=16000, return_tensors="pt"
            )
            transcription_inputs = {k: v.to(device) for k, v in transcription_inputs.items()}
            logits = transcription_model(**transcription_inputs).logits

            inputs = feature_extractor(
                waveform, sampling_rate=16000, return_tensors="pt", padding=True
            )

            input_values = inputs["input_values"].to(device)
            emotion_logits = emotion_model(input_values).logits
            probs = F.softmax(emotion_logits, dim=1)
            predicted_classes = torch.argmax(emotion_logits, dim=1)

            # Extract per-sample stats
            logits_cpu = emotion_logits.cpu()
            max_confidences, _ = torch.max(probs, dim=1)
            full_logits = logits_cpu[0].tolist()
            rounded_logits = [round(x, 2) for x in full_logits]
            second_idx = int(np.argsort(rounded_logits)[-2])
            confidence = round(float(max_confidences[0].cpu()), 2)

            result = {
                "true_label": label,
                "logits": rounded_logits,
                "predicted_label": int(predicted_classes[0]),
                "confidence": confidence,
                "second_highest_index": second_idx
            }

            # Save to CSV
            df = pd.DataFrame([result])
            output_csv = model_path + "_predictions_with_logits_single.csv"
            df.to_csv(output_csv, index=False)
            print(f"Single prediction saved to {output_csv}")

        return result

# Prepare data
data_dir = "CremaTest"
labels, files = [], []
for fname in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, fname)
    if os.path.isfile(path):
        emo = fname.split('_')[2]
        if emo in label_map:
            labels.append(label_map[emo])
            files.append(path)

# Run evaluation
emotionmodel = EmotionModel(files, labels)
emotionmodel.evaluate_and_store_predictions()

# Compute metrics
f1 = f1_score(labels, emotionmodel.actual_predictions, average='macro')
print("Correct prediction:", emotionmodel.correct_prediction)
print("Wrong predictions:", emotionmodel.wrong_prediction)
print("Total predictions:", emotionmodel.total_predictions)
print("Accuracy:", emotionmodel.correct_prediction / emotionmodel.total_predictions)
print("F1 score:", f1)

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
    actor_id = filepath.split('/')[1].split('_')[0]  
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
    actor_id = filepath.split('/')[1].split('_')[0]  
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
    actor_id = filepath.split('/')[1].split('_')[0] 
    intensity = filepath.split('/')[1].split('_')[3]
    if intensity != "XX.wav":
        intensity_count += 1
    if GENDERMAP.get(actor_id) == "Male":
        male_count += 1

print(f"{male_count}/{total_count}")
print(f"Intensity: {intensity_count}/{total_count}")

# Save data for confusion matrix
results_data = {
    "Actual Labels": emotionmodel.labels,
    "Predicted Labels": emotionmodel.actual_predictions
}

df_results = pd.DataFrame(results_data)

df_results.to_csv("WavLMLargeCF.csv", index=False)