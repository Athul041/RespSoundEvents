from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from LungDataSet import LungDataSet

dataset_folder = '../HF_Lung_V1'
train_data_path = os.path.join(dataset_folder, 'train')
test_data_path = os.path.join(dataset_folder, 'test')

SAMPLE_RATE = 4000
ALL_LABELS = ['I', 'D', 'E', 'Rhonchi', 'Wheeze', 'Stridor']


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    print(f'Random seed {seed} has been set.')

class Net_1(nn.Module):
    def __init__(self, input_dimensions, n_hidden, output_dimensions, n_layers=1, drop_prob=0, bidirectional=False, device="cpu"):
        super(Net_1, self).__init__()

        self.n_features = input_dimensions
        self.n_hidden_units = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.bidirectional = bidirectional
        self.n_out = output_dimensions
        self.device = device
        self.directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=self.n_hidden_units,
                            num_layers=self.n_layers,
                            dropout=self.drop_prob,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        # self.dropout = nn.Dropout(self.drop_prob)

        self.dense = nn.Linear(
            in_features=self.directions*self.n_hidden_units, out_features=self.n_out)

        # self.activation = nn.Sigmoid()

        self.network = nn.Sequential(self.lstm,
                                     self.dense)
        self.to(device)

    def forward(self, x):
        h_0 = torch.zeros(self.directions*self.n_layers, x.shape[0], self.n_hidden_units).to(self.device)
        c_0 = torch.zeros(self.directions*self.n_layers, x.shape[0], self.n_hidden_units).to(self.device)
        output, (h_t, c_t) = self.lstm(x, (h_0, c_0))
        output = self.dense(output)
        # output = self.activation(output)

        return output
    
class GRUNet(nn.Module):
    def __init__(self, input_dimensions, n_hidden, output_dimensions, n_layers=1, drop_prob=0, bidirectional=False, device="cpu"):
        super(GRUNet, self).__init__()

        self.n_features = input_dimensions
        self.n_hidden_units = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.bidirectional = bidirectional
        self.n_out = output_dimensions
        self.device = device
        self.directions = 2 if self.bidirectional else 1

        self.gru = nn.GRU(input_size=self.n_features,
                            hidden_size=self.n_hidden_units,
                            num_layers=self.n_layers,
                            batch_first=True,
                            dropout=self.drop_prob,
                            bidirectional=self.bidirectional)

        # self.dropout = nn.Dropout(self.drop_prob)

        self.dense = nn.Linear(
            in_features=self.directions*self.n_hidden_units, out_features=self.n_out)

        # self.activation = nn.Sigmoid()

        self.network = nn.Sequential(self.gru,
                                     self.dense)
        self.to(device)

    def forward(self, x):
        h_0 = torch.zeros(self.directions*self.n_layers, x.shape[0], self.n_hidden_units).to(self.device)
        # c_0 = torch.zeros(self.directions*self.n_layers, x.shape[0], self.n_hidden_units).to(self.device)
        output, h_t = self.gru(x, h_0)
        output = self.dense(output)
        # output = self.activation(output)

        return output


def train_model(num_epochs, model, optimizer, scheduler, loss_fn, train_loader, validation_loader, device):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_losses = []
    training_accuracy = []
    validation_losses = []
    validation_accuracy = []
    output_threshold = 0.5
    best_validation_loss = np.Inf
    for e in range(num_epochs):
        train_batch_loss = []
        validation_batch_loss = []
        train_batch_accuracy = []
        validation_batch_accuracy = []
        # update weights using training data

        model.train(True)
        for train_batch_id, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)
            out = model(data)
            # out = out[-1,:,:].squeeze()
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            pred = out.detach().cpu().numpy() > output_threshold
            accuracy = np.sum(pred == labels.detach().cpu().numpy(
                ))/(labels.shape[0]*labels.shape[1]*labels.shape[2])
            train_batch_loss.append(loss.item())
            train_batch_accuracy.append(accuracy)

        model.train(False)
        with torch.no_grad():
            for val_batch_id, (data, labels) in enumerate(validation_loader):
                data = data.to(device)
                labels = labels.to(device)
                out = model(data)
                # out = out[-1,:,:].squeeze()
                loss = loss_fn(out, labels)
                pred = out.detach().cpu().numpy() > output_threshold
                accuracy = np.sum(pred == labels.cpu().numpy(
                    ))/(labels.shape[0]*labels.shape[1]*labels.shape[2])
                validation_batch_loss.append(loss.item())
                validation_batch_accuracy.append(accuracy)

        scheduler.step(loss)

        training_losses.append(sum(train_batch_loss)/(train_batch_id + 1))
        validation_losses.append(sum(validation_batch_loss)/(val_batch_id + 1))
        training_accuracy.append(
            sum(train_batch_accuracy)/(train_batch_id + 1))
        validation_accuracy.append(
            sum(validation_batch_accuracy)/(val_batch_id + 1))

        # print out epoch metrics
        print('-'*15)

        print("Epoch: {}:".format(e+1))
        print("\t train loss: {:.2f}, validation loss: {:.2f}".format(
            training_losses[-1], validation_losses[-1]))
        print("\t train accuracy: {:.2f}, validation accuracy: {:.2f}".format(
            training_accuracy[-1], validation_accuracy[-1]))

        if best_validation_loss > validation_losses[-1]:
            best_validation_loss = validation_losses[-1]
            model_path = './Models/model_{}_{}.pt'.format(timestamp, e+1)
            torch.save(model.state_dict(), model_path)

    return training_losses, validation_losses, training_accuracy, validation_accuracy


def collate_fn(batch):
    tensors, targets = [], []
    for waveform, label in batch:
        tensors += [waveform.squeeze().t().float()]
        targets += [label]
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    return tensors, targets

def create_dataloader(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    return dataloader


def label_encoder(labels, windows, sample_rate):
    labels_dict = {label: i for i, label in enumerate(ALL_LABELS)}
    labels_array = torch.zeros(windows, len(ALL_LABELS))
    start = 0
    end = 15*sample_rate
    window_size = (end - start) / windows
    row_starts = labels['start'].to_numpy() * sample_rate
    row_ends = labels['end'].to_numpy() * sample_rate
    for i in range(windows):
        win_start = i*window_size + start
        win_end = win_start + window_size
        relevant_rows = (row_starts <= win_start) & (row_ends >= win_end)
        relevant_labels = labels['class'].to_numpy()[relevant_rows]
        label_indices = [labels_dict[label] for label in relevant_labels]
        labels_array[i, label_indices] = 1
    return labels_array


if __name__ == "__main__":
    list_files = []
    data_folder = train_data_path
    for file in os.listdir(data_folder):
        if file.endswith(".wav"):
            list_files.append(os.path.join(
                data_folder, os.path.splitext(file)[0]))

    signal_transforms = []
    window_function = torch.hamming_window
    window_length = 256
    hop_length = 64
    n_bands = 256
    spectrogram = torchaudio.transforms.Spectrogram(
        # sample_rate=SAMPLE_RATE,
        n_fft=n_bands,
        win_length=window_length,
        hop_length=hop_length,
        window_fn=window_function
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=256,
        win_length=window_length,
        hop_length=hop_length,
        window_fn=window_function,
        n_mels=n_bands//2 +1,
        f_min=0,
        f_max=SAMPLE_RATE
    )

    n_mfccs = 60
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=n_mfccs,
        log_mels=True,
        melkwargs={"n_fft": 256, 
                "win_length":window_length,
                "hop_length": hop_length, 
                "n_mels": n_bands//2 + 1,
                "window_fn": window_function, 
                "f_min": 0,
                "f_max": SAMPLE_RATE},
    )
    num_features = 0
    
    signal_transforms.append(spectrogram)
    num_features += (n_bands//2 + 1)
    
    signal_transforms.append(mfcc)
    num_features += n_mfccs

    # signal_transforms.append(mel_spectrogram)
    # num_features += (n_bands//2 + 1)
    
    seed = 0
    set_random_seeds(seed)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    train = LungDataSet(list_files, transformations=signal_transforms,
                        targets_transform=label_encoder)
    train_dataset, validation_dataset = random_split(train, [np.floor(len(
        list_files)*0.9).astype(int), len(list_files) - np.floor(len(list_files)*0.9).astype(int)])

    batch_size = 256

    train_loader = create_dataloader(train_dataset, batch_size)
    validation_loader = create_dataloader(validation_dataset, batch_size)

    num_windows = np.ceil(15*SAMPLE_RATE/hop_length)
    
    hidden_layers = 256
    model = Net_1(input_dimensions=num_features, n_hidden=hidden_layers, output_dimensions=len(
        ALL_LABELS), n_layers=2, drop_prob=0.2, bidirectional=False, device=device)

    learning_rate = 0.0001
    num_epochs = 50
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)

    training_losses, validation_losses, training_accuracy, validation_accuracy = train_model(
        num_epochs, model, optimizer, scheduler, loss_fn, train_loader, validation_loader, device)

    plt.figure(figsize=(4, 6))
    plt.plot(training_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.legend()
    plt.title("Loss")
    plt.show()
    plt.clf()

    plt.figure(figsize=(4, 6))
    plt.plot(training_accuracy, label="train_accuracy")
    plt.plot(validation_accuracy, label="validation_accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.show()
    plt.clf()
