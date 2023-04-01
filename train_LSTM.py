import datetime
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import random
import os
from scipy import signal
import librosa
import matplotlib.pyplot as plt

dataset_folder = '../HF_Lung_V1'
train_data_path = os.path.join(dataset_folder, 'train')
test_data_path = os.path.join(dataset_folder, 'test')

SAMPLE_RATE = 4000
ALL_LABELS = ['I', 'D', 'E', 'Rhonchi', 'Wheeze', 'Stridor']


class LungDataSet(Dataset):
    def __init__(self, file_list, transform, targets_transform):
        self.file_list = file_list
        self.target_transform = targets_transform
        self.target_sample_rate = SAMPLE_RATE
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        labels = self.__get_audio_sample_labels(index)

        # Read Audio from path
        wav, sample_rate = librosa.load(audio_sample_path, sr=None, mono=False)
        cutoff_freq = 80

        # Apply high pass filter cutoff = 80Hz, filter order = 10
        b, a = signal.butter(N=10, Wn=cutoff_freq,
                             btype='high', fs=sample_rate)
        wav_filtered = signal.filtfilt(b, a, wav)

        # Convert the np array to torch tensor and transfer to device
        torch_signal = torch.tensor(wav_filtered.copy()).reshape(1, -1)

        # Resample the signal to 4kHz
        torch_signal = self._resample_if_necessary(torch_signal, sample_rate)

        # Apply transformations
        if self.transform:
            torch_signal = self.transform(torch_signal)
        if self.target_transform:
            labels = self.target_transform(torch_signal, sample_rate, labels)

        return torch_signal, labels

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.file_list[index] + ".wav")
        return path

    def __get_audio_sample_labels(self, index):
        path = os.path.join(self.file_list[index] + "_label.txt")
        labels = pd.read_csv(path, sep=' ', header=None,
                             names=["class", "start", "end"])
        labels['start'] = pd.to_timedelta(labels['start']).dt.total_seconds()
        labels['end'] = pd.to_timedelta(labels['end']).dt.total_seconds()
        return labels

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate)
            signal = resampler(signal)

        return signal


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    print(f'Random seed {seed} has been set.')


class Net(nn.Module):
    def __init__(self, input_dimensions, hidden_layers, output_dimensions, device="cpu"):
        super(Net, self).__init__()
        self.hidden_layers = hidden_layers
        self.num_lstm_layers = len(hidden_layers)
        self.model = nn.Sequential()
        in_num = input_dimensions
        for i in range(self.num_lstm_layers):
            out_num = hidden_layers[i]
            layer = nn.LSTMCell(in_num, out_num)
            self.model.add_module('LSTM_%d' % i, layer)
            # LSTMCell by default has an activation function of tanh in the layer
            in_num = out_num
        # end with a linear layer
        layer = nn.Linear(in_num, output_dimensions)
        self.model.add_module('Linear_end', layer)
        self.device = device
        self.to(device)

    def forward(self, x):
        outputs = []
        # num samples refers to the batch size, not the length of the sequence
        num_samples = x.shape[0]
        # For general NN, the forward function was all about calling the sequential object we defined in the constructor.
        # Now, we need to be more careful about what we feed into the RNN/LSTM layers.
        # Recurrent layers require the latent state of their cells

        # We need to initialize the cell state c,
        # and hidden state h
        # for every layer.
        # pytorch does this would like us to initialize this the same size as the input (num_samples)
        h_t = {}
        c_t = {}
        for i in range(self.num_lstm_layers):
            h_t[i] = torch.zeros(
                num_samples, self.hidden_layers[i], dtype=torch.float32).to(self.device)
            c_t[i] = torch.zeros(
                num_samples, self.hidden_layers[i], dtype=torch.float32).to(self.device)
        # dim = time dimension
        for input_t in x.split(1, dim=2):
            for i in range(self.num_lstm_layers):
                module = self.model._modules['LSTM_%d' % i]
                if i == 0:
                    input_t = input_t.squeeze().to(self.device)
                    h_t[i], c_t[i] = module(input_t, (h_t[i], c_t[i]))
                else:
                    h_t[i], c_t[i] = module(h_t[i-1], (h_t[i], c_t[i]))
            # pass the hidden state along the the linear layer
            module = self.model._modules['Linear_end']
            output = module(h_t[i])
            outputs.append(output)

        # Change outputs to tensor
        outputs = torch.stack(outputs, dim=0)
        return outputs


class Net_1(nn.Module):
    def __init__(self, input_dimensions, n_hidden, output_dimensions, n_layers=1, drop_prob=0, bidirectional=False, device="cpu"):
        super(Net_1, self).__init__()

        self.n_features = input_dimensions
        self.n_hidden_units = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.bidirectional = bidirectional,
        self.n_out = output_dimensions
        self.device = device

        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=self.n_hidden_units,
                            num_layers=self.n_layers,
                            dropout=self.drop_prob,
                            batch_first=True,
                            bidirectional=False)

        # self.dropout = nn.Dropout(self.drop_prob)

        self.dense = nn.Linear(
            in_features=self.n_hidden_units, out_features=self.n_out)

        self.activation = nn.Sigmoid()

        self.network = nn.Sequential(self.lstm,
                                     self.dense,
                                     self.activation)
        self.to(device)

    def forward(self, x):
        # print("x ", x.shape)
        h_0 = torch.zeros(
            self.n_layers, x.shape[0], self.n_hidden_units).to(self.device)
        # print("h_0 ", h_0.shape)
        c_0 = torch.zeros(
            self.n_layers, x.shape[0], self.n_hidden_units).to(self.device)
        output, (h_t, c_t) = self.lstm(x, (h_0, c_0))
        # print("output", output.shape)
        output = self.dense(output)
        output = self.activation(output)

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
        for batch_idx, (data, labels) in enumerate(train_loader):
            # print(data.shape)
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
        for batch_idx, (data, labels) in enumerate(validation_loader):
            data = data.to(device)
            labels = labels.to(device)
            out = model(data)
            # out = out[-1,:,:].squeeze()
            loss = loss_fn(out, labels)
            pred = out.detach().cpu().numpy() > output_threshold
            accuracy = np.sum(pred == labels.detach().cpu().numpy(
            ))/(labels.shape[0]*labels.shape[1]*labels.shape[2])
            validation_batch_loss.append(loss.item())
            validation_batch_accuracy.append(accuracy)

        scheduler.step()

        training_losses.append(sum(train_batch_loss)/(batch_idx + 1))
        validation_losses.append(
            sum(validation_batch_loss)/(batch_idx + 1))
        training_accuracy.append(
            sum(train_batch_accuracy)/(batch_idx + 1))
        validation_accuracy.append(
            sum(validation_batch_accuracy)/(batch_idx + 1))

        # print out epoch metrics
        print('-'*15)

        print("Epoch: {}:".format(e+1))
        print("\t train loss: {:.2f}, validation loss: {:.2f}".format(
            training_losses[-1], validation_losses[-1]))
        print("\t train accuracy: {:.2f}, validation accuracy: {:.2f}".format(
            training_accuracy[-1], validation_accuracy[-1]))
        
        if best_validation_loss < validation_losses[-1]:
            best_validation_loss = validation_losses[-1]
            model_path = 'model_{}_{}'.format(timestamp, e)
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


def create_dataloader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=6, pin_memory=True, collate_fn=collate_fn)
    return train_dataloader


def label_encoder(torch_signal, sample_rate, labels):
    windows = torch_signal.shape[-1]
    labels_dict = {label: i for i, label in enumerate(ALL_LABELS)}
    labels_array = torch.zeros(windows, len(ALL_LABELS))
    start = 0
    end = 15*sample_rate
    window_size = (end - start) / windows
    row_starts = labels['start'] * sample_rate
    row_ends = labels['end'] * sample_rate
    for i in range(windows):
        win_start = i*window_size + start
        win_end = win_start + window_size
        relevant_rows = (row_starts <= win_start) & (row_ends >= win_end)
        relevant_labels = labels.loc[relevant_rows, 'class']
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

    window_function = torch.hamming_window
    window_length = 256
    hop_length = 64
    n_bands = 256
    spectrogram = torchaudio.transforms.Spectrogram(
        # sample_rate=SAMPLE_RATE,
        n_fft=256,
        win_length=window_length,
        hop_length=hop_length,
        window_fn=window_function
    )

    seed = 0
    set_random_seeds(seed)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    dataset = LungDataSet(list_files, transform=spectrogram,
                          targets_transform=label_encoder)

    train_dataset = Subset(dataset=dataset, indices=range(
        0, np.floor(len(dataset)*0.8).astype(int)))
    test_dataset = Subset(dataset=dataset, indices=range(
        np.floor(len(dataset)*0.8).astype(int), len(dataset)))

    batch_size = 256

    train_loader = create_dataloader(train_dataset, batch_size)
    test_loader = create_dataloader(test_dataset, batch_size)

    # net = Net(num_recurrent_layers=1, num_input_features=201,
    #           hidden_layer_nodes=32, output_feature_nums=len(ALL_LABELS))

    num_windows = np.ceil(15*SAMPLE_RATE/hop_length)
    num_features = n_bands//2 + 1
    hidden_layers = 64
    model = Net_1(input_dimensions=num_features, n_hidden=hidden_layers, output_dimensions=len(
        ALL_LABELS), n_layers=1, drop_prob=0, bidirectional=False, device=device)
    learning_rate = 0.01
    num_epochs = 10
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    training_losses, testing_losses, training_accuracy, testing_accuracy = trainingLayerWise(
        num_epochs, model, optimizer, scheduler, loss_fn, train_loader, test_loader, device)

    plt.figure(figsize=(4, 6))
    plt.plot(training_losses, label="train_loss")
    plt.plot(testing_losses, label="test_loss")
    plt.legend()
    plt.title("loss")
    plt.show()
    plt.clf()

    plt.figure(figsize=(4, 6))
    plt.plot(training_accuracy, label="train_accuracy")
    plt.plot(testing_accuracy, label="test_accuacy")
    plt.legend()
    plt.title("accuracy")
    plt.show()
    plt.clf()
