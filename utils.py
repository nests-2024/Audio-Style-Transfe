import copy
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from NeuralStyleTransfer import GramMatrix, StyleLoss

def read_audio_spectum(filename, duration=16, n_fft=2048):
    x, fs = librosa.load(filename, duration=duration)
    S = librosa.stft(x, n_fft=n_fft)
    S = np.log1p(np.abs(S))  
    return S, fs


def plot_spectrum(spectrum):
    spec_db = librosa.amplitude_to_db(spectrum, ref=np.max)
    librosa.display.specshow(spec_db)
    plt.show()


def get_style_model_and_losses(cnn, style_float, style_weight, style_layers):
    cnn = copy.deepcopy(cnn)
    style_losses = []
    model = nn.Sequential()
    gram = GramMatrix()
    if torch.cuda.is_available():
        model = model.cuda()
        gram = gram.cuda()

    name = 'conv_1'
    model.add_module(name, cnn.cnn1)

    if name in style_layers:
        target_feature = model(style_float).clone()
        target_feature_gram = gram(target_feature)
        style_loss = StyleLoss(target_feature_gram, style_weight)
        model.add_module("style_loss_1", style_loss)
        style_losses.append(style_loss)

    return model, style_losses


def get_input_param_optimizer(input_float, lr):
    input_param = nn.Parameter(input_float.data)
    optimizer = optim.Adam([input_param], lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    return input_param, optimizer


def run_style_transfer(cnn, style_float, input_float, num_steps=2500, style_weight=2500, style_layers=['conv_1'], lr=0.03):
    print('Building the style transfer model..')
    model, style_losses = get_style_model_and_losses(cnn, style_float, style_weight, style_layers)
    input_param, optimizer = get_input_param_optimizer(input_float, lr)
    print('Optimizing..')
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0

            for sl in style_losses:
                style_score += sl.backward()

            run[0] += 1
            if run[0] % 100 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:8f}'.format(style_score.item()))
                print()
            return style_score

        optimizer.step(closure)
    input_param.data.clamp_(0, 1)
    return input_param.data
