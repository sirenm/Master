import os
import shutil
import torchaudio
from torchaudio.transforms import PitchShift

def copy_files(source_folder, filepath):
        shutil.copy(os.path.join(source_folder), filepath)

def DataAugmentation_PitchShifting():
        os.makedirs("CremaTrainPitchShifting", exist_ok=True)
        for file in os.listdir("CremaTrain"):
            fileinfo_list = file.split("_")
            if len(fileinfo_list) == 4:
                waveform, sample_rate = torchaudio.load(os.path.join("CremaTrain", file))
                pitchshift = PitchShift(sample_rate=sample_rate, n_steps=0.6)
                pitch_shifted = pitchshift(waveform)
                pitchshift_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_PitchShifting.wav"
                torchaudio.save(os.path.join("CremaTrainPitchShifting", pitchshift_file), pitch_shifted, sample_rate)
                copy_files(os.path.join("CremaTrain", file), os.path.join("CremaTrainPitchShifting", file))

DataAugmentation_PitchShifting()
