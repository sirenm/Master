import pandas as pd
import numpy as np
import os
import librosa
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class SpeechEmotionRecognition:
    def __init__(self, dataset_name, model_name="facebook/wav2vec2-base"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.df = None
        self.label_map = None
        self.feature_extractor = None
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None        
        self.y_val = None
        self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_dataset(self):
        # Download the dataset
        dataset_path = "Crema"
        self.df = self.read_data(dataset_path)
        print('Dataset is Loaded', len(self.df))
        print(self.df.head(5))
        print(self.df['label'].value_counts())
        

    def read_data(self, path):
        paths, labels = [], []
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                label = filename.split('_')
                if len(label) == 4:
                    paths.append(os.path.join(dirname, filename))
                    labels.append(label[2].lower())
        return pd.DataFrame({'speech': paths, 'label': labels})

    def encode_labels(self):
        # Encode labels
        self.label_map = {label: idx for idx, label in enumerate(self.df['label'].unique())}
        self.df['label'] = self.df['label'].map(self.label_map)
        print(self.df.head(5))

    def initialize_feature_extractor(self):
        # Initialize Wav2Vec2 feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.to(device)

    def extract_features(self, audio_path):
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        # Pass through Wav2Vec2 model
        with torch.no_grad():
            outputs = self.model(inputs["input_values"])


        # Get mean across time dimension for fixed-size vector
        features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return features

    def extract_all_features(self):
        # Extract features for all audio files
        self.X = np.array([self.extract_features(path) for path in self.df['speech']])
        self.y = self.df['label'].values

    def split_data(self):
        # Split dataset into training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Check class distributions
        print("Train set label distribution:\n", self.y_train.value_counts())
        print("\nTest set label distribution:\n", self.y_test.value_counts())

            
    def split_data_stratifiedShuffleSplit(self):        
        # Define split: 1 split, 80% train, 20% test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in splitter.split(self.X, self.y):
            train_df = self.df.iloc[train_idx].reset_index(drop=True)
            test_df = self.df.iloc[test_idx].reset_index(drop=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_df['speech'], train_df['label'],test_df['speech'], test_df['label']
        
        # Check class distributions
        print("Train set label distribution:\n", train_df['label'].value_counts())
        print("\nTest set label distribution:\n", test_df['label'].value_counts())

    def split_data_stratifiedShuffleSplit_dev(self):        
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Further split training into 70% train and 30% validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.3, random_state=42, stratify=y_train_val
        )

        print("Train set label distribution:\n", pd.Series(self.y_train).value_counts())
        print("\nTest set label distribution:\n", pd.Series(self.y_test).value_counts())
        print("\nValidation set label distribution:\n", pd.Series(self.y_val).value_counts())
        
    def cross_validate_model(self):
        # Perform 5-Fold Stratified Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.rf_clf, self.X_train, self.y_train, cv=skf, scoring='accuracy')

        print("Cross-Validation Accuracy Scores:", scores)
        print("Mean Accuracy:", np.mean(scores))

    def train_rf_classifier(self):
        # Train Random Forest Classifier
        
        self.rf_clf.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Evaluate on Validation Set
        y_val_pred = self.rf_clf.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        print("Validation Accuracy:", val_accuracy)
        print("Validation Classification Report:\n", classification_report(self.y_val, y_val_pred))

        # Evaluate on Test Set
        y_test_pred = self.rf_clf.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print("Test Accuracy:", test_accuracy)
        print("Test Classification Report:\n", classification_report(self.y_test, y_test_pred))

        


# Using the class to execute the process
if __name__ == "__main__":
    emotion_recognition = SpeechEmotionRecognition(dataset_name="Crema")
    
    # Load dataset and preprocess
    emotion_recognition.load_dataset()
    emotion_recognition.encode_labels()
    emotion_recognition.initialize_feature_extractor()
    
    # Feature extraction and model training
    emotion_recognition.extract_all_features()
    emotion_recognition.split_data_stratifiedShuffleSplit_dev()
    
    # Perform Cross-Validation
    emotion_recognition.cross_validate_model()
    
     # Train and Evaluate Model
    emotion_recognition.train_rf_classifier()
    emotion_recognition.evaluate_model()
