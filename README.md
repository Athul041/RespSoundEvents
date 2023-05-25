# RespSoundEvents
Identifying and labelling events occuring in the Lungs from respiratory audio using different RNN Models.

## Objectives
This project aims to develop an automated system using deep recurrent neural network models to accurately identify and label respiratory sound events from audio signals captured during respiratory auscultation. By utilizing time-frequency spectral and cepstral features, the proposed approach seeks to overcome the limitations of existing methods, enabling real-time monitoring and providing deeper medical insights into patients' respiratory health.

The study builds upon the principles of automatic sound event detection, commonly applied in speech recognition and sound event detection domains. By adapting these techniques to respiratory auscultation audio signals, the deep recurrent neural network model will be trained to categorize lung sounds into normal breathing sounds and adventitious sounds, such as crackles and wheezes. The temporal nature of the respiratory events will be captured, enabling more accurate and detailed analysis of a patient's condition.

## Dataset
The [HF Lung V1 Database](https://gitlab.com/techsupportHF/HF_Lung_V1) is the largest publicly available respiratory sound dataset. The dataset consists of breath sound recordings that were collected using an electronic stethoscope and a customized acoustic recording device with eight microphones attached to various locations of the chest. There are 9765 recordings, each 15 seconds long captured from 279 patients. Each 15-second sample is generated by combining the recording from 8 different parts of the patient’s chest, effectively making it a single-channel audio sample. The authors provide six labels:
- I - Inhalation (34095)
- E - Exhalation (18349)
- W - wheezes (8457)
- S - stridor (686)
- R - rhonchi (4740)
- D - discontinuos events or Crackles (15606)

## Project Setup
Since lung sounds can be polyphonic, i.e, multiple sound events can be active at a given timeframe, this effectively can be translated to a Multiclass, Multilabel classification problem. The ideal model should however, predict
For each 15s audio file, three different feature sets were extracted from different combinations of spectral and cepstral feature transformations were used to evaluate the RNN models
- 129 - Spectrogram features
- 189 - 129 Spectrogram bands + 60 MFCCs
- 129 - Mel Spectrogram features

All three feature sets were compared by the performance (model fit, accuracy, Recall and Precision) of an LSTM network with the following configuration:
- 2 LSTM layers x 256 nodes (dropout probability: 0.5)
- 1 fully connected dense layer (in: 256, out: 6)
- Sigmoid activation function - since each class is independant of the other it can be seen as multiple binary classifications where each output neuron provides the probability of that class being active in the given time frame.
- Binary Cross Entropy Loss activation function - since each neuron in the sigmoid level acts as a binary class probability.
- 50 epochs
- 0.001 learning rate 
- Adam optimizer with ReduceLROnPlateue with gamma=0.1 over 10 epochs

A BiLSTM model and a GRU model were also used for comparison to the LSTM models.

## Results
