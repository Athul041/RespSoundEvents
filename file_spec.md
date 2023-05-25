# RespSoundEvents
Sound event detection in respiratory sounds using an LSTM
Files:
1. LungDataSet: Custom pytorch dataset which uses preloading input audio into memory for faster data loading. This is more effective when num_workers in dataloader is high
2. train_LSTM: Training script which requires LungDataSet
3. test_model: Testing script which requires models from train_LSTM
4. LSTM_GRU_modelling: ipynb used for training on Google Colab (Note: dataset is not preloaded into RAM due to memory constraints in Colab). 
    Due to restrictions of Colab, cannot use num_workers > 0 in dataloader. However, this overhead was found to be minimal given the performance gains of GPUs in Colab.
5. testing_models: ipynb used for testing all models generated.
7. Dataset used: https://gitlab.com/techsupportHF/HF_Lung_V1/-/blob/master/LICENSE
8. Checkpoint models: best models at the end of training various hyperparameters
9. Figures: Figures generated in all training and testing