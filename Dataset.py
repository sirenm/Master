import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

CREMAFIlECOUNT = 7442

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
    
class DatasetCrema:
    def __init__(self, dataset_filepath):
        self.dataset_filepath = dataset_filepath
        self.train_folder = "CremaTrain"
        self.test_folder = "CremaTest"
        self.eval_folder = "CremaValidation"
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(self.eval_folder, exist_ok=True)

    def copy_files(self, source_folder, filepath):
        shutil.copy(os.path.join(source_folder), filepath)

    def divide_dataset(self):
        file_info = []
        for file in os.listdir(self.dataset_filepath):
            fileinfo_list = file.split("_")
            if len(fileinfo_list) == 4:
                file_info.append({
                        "filename": file,
                        "actor_id": fileinfo_list[0],
                        "statement": fileinfo_list[1],
                        "emotion": fileinfo_list[2],
                        "intensity": "XX.wav" if fileinfo_list[3] == "X.wav" else fileinfo_list[3],
                        "gender": GENDERMAP[fileinfo_list[0]]
                    })
        df = pd.DataFrame(file_info)
        df["balanced"] = df["gender"] + "_" + df["statement"] + "_" + df["emotion"] + "_" + df["intensity"]
        train, rest = train_test_split(df, train_size=4466, stratify=df["balanced"], random_state=42)
        eval, test = train_test_split(rest, train_size=1488, stratify=rest["balanced"], random_state=42)

        for file in train["filename"]:
            self.copy_files(os.path.join(self.dataset_filepath, file), os.path.join(self.train_folder, file))
        for file in eval["filename"]:
            self.copy_files(os.path.join(self.dataset_filepath, file), os.path.join(self.eval_folder, file))
        for file in test["filename"]:
            self.copy_files(os.path.join(self.dataset_filepath, file), os.path.join(self.test_folder, file))



dataset_instance = DatasetCrema("Crema")
print(dataset_instance.divide_dataset())