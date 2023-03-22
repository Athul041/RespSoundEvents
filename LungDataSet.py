import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from scipy import signal
import librosa

dataset_folder = '../HF_Lung_V1'
train_data_path = os.path.join(dataset_folder, 'train')
test_data_path = os.path.join(dataset_folder, 'test')

SAMPLE_RATE = 4000

class LungDataSet(Dataset):
    def __init__(self, file_list, transform, targets_transform):
        self.file_list = file_list
        self.transform = transform
        self.target_transform = targets_transform
        self.target_sample_rate = SAMPLE_RATE
    
    def __len__(self):
        len(self.file_list)
        
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        labels = self.__get_audio_sample_labels(index)
        
        # Read Audio from path
        wav, sample_rate = librosa.load(audio_sample_path, sr=None, mono=False)
        cutoff_freq = 15
        
        # Apply high pass filter cutoff = 80Hz, filter order = 10
        b, a = signal.butter(N=10, Wn=cutoff_freq, btype='high', fs=sample_rate)
        wav_filtered = signal.filtfilt(b, a, wav)
        # Convert the np array to torch tensor
        torch_signal = torch.reshape(wav_filtered, (1,wav_filtered.shape[0]))
        
        # Resample the signal to 4kHz
        torch_signal = self._resample_if_necessary(torch_signal)
        
        # Apply transformations
        if self.transform:
            torch_signal = self.transform(torch_signal)
        if self.target_transform:
            labels = self.target_transform(labels)
            
        return torch_signal, labels
    
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.file_list[index] + ".wav")
        return path
    
    def __get_audio_sample_labels(self, index):
        path = os.path.join(self.file_list[index] + "_label.txt")
        labels = pd.read_csv(path, sep=' ', header=None, names=["class", "start", "end"])
        return labels
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            
        return signal
    
    
if __name__ == "__main__":
    
    data_folder = train_data_path
    
    # Load list of data files
    list_files = []
    for file in os.listdir(data_folder):
        if file.endswith(".wav"):
            list_files.append(os.path.join(data_folder, os.path.splitext(file)[0]))
    
    # Define transform/feature extranction
    window_function = torch.hann_window
    spectrogram = torchaudio.transforms.Spectrogram(
        # sample_rate=SAMPLE_RATE,
        # n_fft=1024,
        win_length=256,
        hop_length=64,
        window_fn=window_function
    )
    
    def label_encoder(torch_signal, labels):
        windows = torch_signal.shape[-1]
        labels_array = np.zeros([windows, len(ALL_LABELS)])
        start = min(labels['start'])*1000
        end = max(labels['end'])*1000
        window_size = np.floor(end - start/windows)
        for i in range(windows):
            win_start = i*window_size + start
            win_end = win_start + window_size
            for index, row in labels.iterrows():
                row_start = row['start']*1000
                row_end = row['end']*1000
                if (row_start <= win_start) and (row_end >= win_end):
                    labels_array[i][ALL_LABELS.index(row['class'])] = 1
        return labels_array
    
    lds = LungDataSet(list_files, spectrogram, label_encoder)