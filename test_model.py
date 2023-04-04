import random
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
import torch
import torchaudio
import os
from torch.utils.data import DataLoader
from LungDataSet import LungDataSet
from train_LSTM import Net_1

# MODEL_PATH = './model_{}_{}'
dataset_folder = '../HF_Lung_V1'
test_data_path = os.path.join(dataset_folder, 'test')
SAMPLE_RATE = 4000
ALL_LABELS = ['I', 'D', 'E', 'Rhonchi', 'Wheeze', 'Stridor']

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

def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    print(f'Random seed {seed} has been set.')

def test_model(model, test_loader, device):
    testing_accuracy = []
    output_threshold = 0.5
    y_pred = []
    y_true = []
    model.eval()
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        out = model(data)
        pred = out.detach().cpu().numpy() > output_threshold
        
    # Build confusion matrix
    cf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in ALL_LABELS],
                        columns = [i for i in ALL_LABELS])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')     
    
    return cf_matrix


    
if __name__ == "__main__":
    seed = 0
    set_random_seeds(seed)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    
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
    
    list_files = []
    data_folder = test_data_path
    for file in os.listdir(data_folder):
        if file.endswith(".wav"):
            list_files.append(os.path.join(
                data_folder, os.path.splitext(file)[0]))

    train = LungDataSet(list_files, transform=spectrogram,
                      targets_transform=label_encoder)

    num_windows = np.ceil(15*SAMPLE_RATE/hop_length)
    num_features = n_bands//2 + 1
    hidden_layers = 64
    
    saved_model = Net_1(input_dimensions=num_features, n_hidden=hidden_layers, output_dimensions=len(
        ALL_LABELS), n_layers=1, drop_prob=0, bidirectional=False, device=device)
    saved_model.load_state_dict(torch.load(MODEL_PATH))
    
    test_dataset = LungDataSet(list_files, transform=spectrogram, 
                        targets_transform=label_encoder)
    
    batch_size = 256

    test_loader = create_dataloader(test_dataset, batch_size)
    
    conf_matrix = test_model(saved_model, test_loader)
    
    
    
    
