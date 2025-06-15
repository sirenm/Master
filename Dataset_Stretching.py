import os
import shutil
import torchaudio
import torchaudio.transforms as T

def copy_files(source_folder, filepath):
        shutil.copy(os.path.join(source_folder), filepath)

def DataAugmentation_Stretching():
        os.makedirs("CremaTrainStretching", exist_ok=True)
        for file in os.listdir("CremaTrain"):
            fileinfo_list = file.split("_")
            if len(fileinfo_list) == 4:
                waveform, sample_rate = torchaudio.load(os.path.join("CremaTrain", file))
                spectrogram = T.Spectrogram(n_fft=1024, hop_length=256, power=None)
                spec = spectrogram(waveform)
                stretch = T.TimeStretch(hop_length=256, n_freq=1024 // 2 + 1)
                stretched_spec = stretch(spec, 0.7)
                inverse_spec = T.InverseSpectrogram(n_fft=1024, hop_length=256)
                stretched_waveform = inverse_spec(stretched_spec)
                stretch_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_Stretched.wav"
                torchaudio.save(os.path.join("CremaTrainStretching", stretch_file), stretched_waveform, sample_rate)
                copy_files(os.path.join("CremaTrain", file), os.path.join("CremaTrainStretching", file))

dataset_instance = DataAugmentation_Stretching()