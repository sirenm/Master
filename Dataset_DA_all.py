import os
import shutil
import torchaudio
import torch
import torchaudio.transforms as T

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

def DataAugmentation_All():
        os.makedirs("CremaTrainDA", exist_ok=True)
        for file in os.listdir("CremaTrain"):
            fileinfo_list = file.split("_")
            if len(fileinfo_list) == 4:
                waveform, sample_rate = torchaudio.load(os.path.join("CremaTrain", file))

                awgn_waveform= add_awgn(waveform, 0.02)
                awgn_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_AWGN_002.wav"
                torchaudio.save(os.path.join("CremaTrainDA", awgn_file), awgn_waveform, sample_rate)

                awgn_waveform= add_awgn(waveform, 0.025)
                awgn_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_AWGN_0025.wav"
                torchaudio.save(os.path.join("CremaTrainDA", awgn_file), awgn_waveform, sample_rate)
                
                spectrogram = T.Spectrogram(n_fft=1024, hop_length=256, power=None)
                spec = spectrogram(waveform)
                stretch = T.TimeStretch(hop_length=256, n_freq=1024 // 2 + 1)
                stretched_spec = stretch(spec, 0.7)
                inverse_spec = T.InverseSpectrogram(n_fft=1024, hop_length=256)
                stretched_waveform = inverse_spec(stretched_spec)
                stretch_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_Stretched_07.wav"
                torchaudio.save(os.path.join("CremaTrainDA", stretch_file), stretched_waveform, sample_rate)

                spectrogram = T.Spectrogram(n_fft=1024, hop_length=256, power=None)
                spec = spectrogram(waveform)
                stretch = T.TimeStretch(hop_length=256, n_freq=1024 // 2 + 1)
                stretched_spec = stretch(spec, 0.8)
                inverse_spec = T.InverseSpectrogram(n_fft=1024, hop_length=256)
                stretched_waveform = inverse_spec(stretched_spec)
                stretch_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_Stretched_08.wav"
                torchaudio.save(os.path.join("CremaTrainDA", stretch_file), stretched_waveform, sample_rate)

                pitchshift = T.PitchShift(sample_rate=sample_rate, n_steps=0.6)
                pitch_shifted = pitchshift(waveform)
                pitchshift_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_PitchShifting_06.wav"
                torchaudio.save(os.path.join("CremaTrainDA", pitchshift_file), pitch_shifted, sample_rate)

                pitchshift = T.PitchShift(sample_rate=sample_rate, n_steps=0.7)
                pitch_shifted = pitchshift(waveform)
                pitchshift_file = f"{fileinfo_list[0]}_{fileinfo_list[1]}_{fileinfo_list[2]}_{fileinfo_list[3].split('.wav')[0]}_PitchShifting_07.wav"
                torchaudio.save(os.path.join("CremaTrainDA", pitchshift_file), pitch_shifted, sample_rate)

                copy_files(os.path.join("CremaTrain", file), os.path.join("CremaTrainDA", file))

DataAugmentation_All()
