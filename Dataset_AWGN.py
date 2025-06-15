import os
import shutil
import torchaudio
import torch

def add_awgn(waveform, snr_db, factor=0.7):
    signal_power = waveform.pow(2).mean()
    signal_power_db = 10 * torch.log10(signal_power)
    noise_power_db = signal_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)

    noise = torch.randn_like(waveform) * torch.sqrt(noise_power) * factor

    noisy_waveform = waveform + noise
    return noisy_waveform

def copy_files(source_folder, filepath):
        shutil.copy(os.path.join(source_folder), filepath)

def DataAugmentation_AWGN():
        os.makedirs("CremaTrainAWGN", exist_ok=True)
        for file in os.listdir("CremaTrain"):
            fileinfo_list = file.split("_")
            if len(fileinfo_list) == 4:
                waveform, sample_rate = torchaudio.load(os.path.join("CremaTrain", file))
                awgn_waveform= add_awgn(waveform, 0.02)
                awgn_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_AWGN.wav"
                torchaudio.save(os.path.join("CremaTrainAWGN", awgn_file), awgn_waveform, sample_rate)
                copy_files(os.path.join("CremaTrain", file), os.path.join("CremaTrainAWGN", file))

DataAugmentation_AWGN()
