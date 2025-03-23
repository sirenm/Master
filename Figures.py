import os
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
from sklearn.metrics import confusion_matrix


DATASET = "Crema"
TESTSET= "CremaTest"
TRAINSET = "CremaTrain"
EVALUATIONSET = "CremaEvaluation"

print("aifc module is available!")

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

SENTENCEMAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes"
}

class PlotFigure:
    def __init__(self, labels, values):
        self.labels = labels
        self.values = values

    def generate_pie_chart(self, titles, imagetitle):
        labels = self.labels
        values = self.values

        fig = make_subplots(rows=1, cols=4, subplot_titles=titles,
                    specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}]])
        
        for index in range(len(self.labels)):
            print(index+1)
            fig.add_trace(go.Pie(labels=labels[index], values=values[index], name=titles[index]), row=1, col=1+index)
            fig.update_layout(title_text=imagetitle, showlegend=True)

        fig.show()

""" def plot_emotion(datasetpath):
    dictionary = dict()
    for file in os.listdir(datasetpath):
        full_path = os.path.join(datasetpath, file)
        if os.path.isfile(full_path):
            file_name_list = file.split('_')
            if len(file_name_list) == 4:
                emotion = file_name_list[2]
                if emotion in dictionary:
                    dictionary[emotion] += 1
                else:
                    dictionary[emotion] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_actor(datasetpath):
    dictionary = dict()
    for file in os.listdir(datasetpath):
        full_path = os.path.join(datasetpath, file)
        if os.path.isfile(full_path):
            file_name_list = file.split('_')
            actor = file_name_list[0]
            if actor in dictionary:
                dictionary[actor] += 1
            else:
                dictionary[actor] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_gender(datasetpath):
    dictionary = dict()
    for file in os.listdir(datasetpath):
        full_path = os.path.join(datasetpath, file)
        if os.path.isfile(full_path):
            file_name_list = file.split('_')
            actor = file_name_list[0]
            gender = GENDERMAP[actor]
            if gender in dictionary:
                dictionary[gender] += 1
            else:
                dictionary[gender] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_statement(datasetpath):
    dictionary = dict()
    for file in os.listdir(datasetpath):
        full_path = os.path.join(datasetpath, file)
        if os.path.isfile(full_path):
            file_name_list = file.split('_')
            statement_code = file_name_list[1]
            statement = SENTENCEMAP[statement_code]
            if statement in dictionary:
                dictionary[statement] += 1
            else:
                dictionary[statement] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_intensity(datasetpath):
    dictionary = dict()
    for file in os.listdir(datasetpath):
        full_path = os.path.join(datasetpath, file)
        if os.path.isfile(full_path):
            file_name_list = file.split('_')
            intensity = file_name_list[3]
            if intensity in dictionary:
                dictionary[intensity] += 1
            else:
                dictionary[intensity] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values """

""" def plot_emotion(datasetpath):
    dictionary = dict()
    df = pd.read_csv(datasetpath)
    for index, value in df.iterrows():
        if value["emotion"] in dictionary:
            dictionary[value["emotion"]] += 1
        else:
            dictionary[value["emotion"]] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_actor(datasetpath):
    dictionary = dict()
    df = pd.read_csv(datasetpath)
    for index, value in df.iterrows():
        if value["filename"].split("/")[-1].split("_")[0] in dictionary:
            dictionary[value["filename"].split("/")[-1].split("_")[0]] += 1
        else:
            dictionary[value["filename"].split("/")[-1].split("_")[0]] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_gender(datasetpath):
    dictionary = dict()
    df = pd.read_csv(datasetpath)
    for index, value in df.iterrows():
        file_name_list = value["filename"].split("/")[-1].split("_")[0]
        gender = GENDERMAP[file_name_list]
        if gender in dictionary:
            dictionary[gender] += 1
        else:
            dictionary[gender] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_statement(datasetpath):
    dictionary = dict()
    df = pd.read_csv(datasetpath)
    for index, value in df.iterrows():
        if value["filename"].split("/")[-1].split("_")[1] in dictionary:
            dictionary[value["filename"].split("/")[-1].split("_")[1]] += 1
        else:
            dictionary[value["filename"].split("/")[-1].split("_")[1]] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values

def plot_intensity(datasetpath):
    dictionary = dict()
    df = pd.read_csv(datasetpath)
    for index, value in df.iterrows():
        if value["filename"].split("/")[-1].split("_")[3] in dictionary:
            dictionary[value["filename"].split("/")[-1].split("_")[3]] += 1
        else:
            dictionary[value["filename"].split("/")[-1].split("_")[3]] = 1
    labels = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]
    return labels, values


labels_data, values_data = plot_emotion("dataset.csv")
labels_train, values_train = plot_emotion("train.csv")
labels_test, values_test = plot_emotion("test.csv")
labels_evaluation, values_evaluation = plot_emotion("eval.csv")
all_labels = []
all_values = []
all_labels.append(labels_data)
all_labels.append(labels_train)
all_labels.append(labels_test)
all_labels.append(labels_evaluation)
all_values.append(values_data)
all_values.append(values_train)
all_values.append(values_test)
all_values.append(values_evaluation)

plotter = PlotFigure(all_labels, all_values)
plotter.generate_pie_chart(["Entire dataset", "Train dataset", "Test dataset", "Evalutation dataset"], "Emotions")

labels_data, values_data = plot_gender("dataset.csv")
labels_train, values_train = plot_gender("train.csv")
labels_test, values_test = plot_gender("test.csv")
labels_evaluation, values_evaluation = plot_gender("eval.csv")
all_labels = []
all_values = []
all_labels.append(labels_data)
all_labels.append(labels_train)
all_labels.append(labels_test)
all_labels.append(labels_evaluation)
all_values.append(values_data)
all_values.append(values_train)
all_values.append(values_test)
all_values.append(values_evaluation)

plotter = PlotFigure(all_labels, all_values)
plotter.generate_pie_chart(["Entire dataset", "Train dataset", "Test dataset", "Evaluation dataset"], "Genders")


labels_data, values_data = plot_statement("dataset.csv")
labels_train, values_train = plot_statement("train.csv")
labels_test, values_test = plot_statement("test.csv")
labels_evaluation, values_evaluation = plot_statement("eval.csv")
all_labels = []
all_values = []
all_labels.append(labels_data)
all_labels.append(labels_train)
all_labels.append(labels_test)
all_labels.append(labels_evaluation)
all_values.append(values_data)
all_values.append(values_train)
all_values.append(values_test)
all_values.append(values_evaluation)

plotter = PlotFigure(all_labels, all_values)
plotter.generate_pie_chart(["Entire dataset", "Train dataset","Test dataset", "Evaluation dataset"], "Statements")

labels_data, values_data = plot_intensity("dataset.csv")
labels_train, values_train = plot_intensity("train.csv")
labels_test, values_test = plot_intensity("test.csv")
labels_evaluation, values_evaluation = plot_intensity("eval.csv")
all_labels = []
all_values = []
all_labels.append(labels_data)
all_labels.append(labels_train)
all_labels.append(labels_test)
all_labels.append(labels_evaluation)
all_values.append(values_data)
all_values.append(values_train)
all_values.append(values_test)
all_values.append(values_evaluation)

plotter = PlotFigure(all_labels, all_values)
plotter.generate_pie_chart(["Entire dataset", "Train dataset", "Test dataset", "Evaluation dataset"], "Intensities")


all_labels = [["Correct", "Wrong"]]
all_values = [[1108, 380],[2610, 1111]]

plotter = PlotFigure(all_labels, all_values)
plotter.generate_pie_chart(["16 batch | 10 epochs | 0.745301941999375 F1-score", "16 batch | 10 epochs |  0.7001205150795126 F1-score"], "Results")

plottermap = ["Total", "XX.wav", "LO.wav", "MD.wav", "HI.wav"]
def plot_statements_intesity(datasetpath):
    dictionary = dict()
    for file in os.listdir(datasetpath):
        full_path = os.path.join(datasetpath, file)
        if os.path.isfile(full_path):
            file_name_list = file.split('_')
            statement = file_name_list[1]
            intensity = file_name_list[3]
            if statement in dictionary:
                if intensity == "X.wav":
                    intensity = "XX.wav"
                index = plottermap.index(intensity)
                dictionary[statement][index] += 1
                dictionary[statement][0] += 1
            else:
                dictionary[statement] = [0] * 5
                index = plottermap.index(intensity)
                dictionary[statement][index] = 1
                dictionary[statement][0] = 1
    data = {
    'Category': [key for key in dictionary.keys() for _ in range(5)],
    'Intensity': ["Total", "XX.wav", "LO.wav", "MD.wav", "HI.wav"] * len(dictionary),
    'Count': [value[i] for _, value in dictionary.items() for i in range(5)]
    }
    return data

df_total = pd.DataFrame(plot_statements_intesity(DATASET))
df_train = pd.DataFrame(plot_statements_intesity(TRAINSET))
df_test = pd.DataFrame(plot_statements_intesity(TESTSET))
df_eval = pd.DataFrame(plot_statements_intesity(EVALUATIONSET))

fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=["Total", "Trainset", "Testset", "Evaluationset"]
)

datasets = [df_total, df_train, df_test, df_eval]
titles = ["Total", "Trainset", "Testset", "Evaluationset"]

for i, (df, title) in enumerate(zip(datasets, titles)):
    row = (i // 2) + 1 
    col = (i % 2) + 1   
    for intensity in plottermap:
        subset = df[df["Intensity"] == intensity]
        fig.add_trace(
            go.Bar(
                x=subset["Category"], 
                y=subset["Count"], 
                name=intensity, 
                legendgroup=intensity
            ),
            row=row, col=col
        )

fig.update_layout(
    title_text="Intensity Distribution per Category",
    barmode='group',
    showlegend=True,
    width=1200, 
    height=800 
)

fig.show()  """


""" label_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "SAD": 3,
    "HAP": 4,
    "NEU": 5
}

file_path = 'results.csv'
df = pd.read_csv(file_path)
y_pred_labels = df['Predicted Labels']
y_test_labels = df['Actual Labels']

cm = confusion_matrix(y_test_labels, y_pred_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='d')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show() """

label_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "SAD": 3,
    "HAP": 4,
    "NEU": 5
}

emotions = [emotion for emotion, _ in sorted(label_map.items(), key=lambda item: item[1])]

# Load data and compute confusion matrix
file_path = 'results.csv'
df = pd.read_csv(file_path)
y_pred_labels = df['Predicted Labels']
y_test_labels = df['Actual Labels']

cm = confusion_matrix(y_test_labels, y_pred_labels)

# Plot the confusion matrix with emotion labels
plt.figure(figsize=(12, 10))
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='d',
            xticklabels=emotions, yticklabels=emotions)
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()