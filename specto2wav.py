import torch
from torch.autograd import Variable
import torchaudio
import torchaudio.functional as F

import torchaudio.transforms as T
from pytorch_musicnet.musicnet import MusicNet
import matplotlib.pyplot as plt
import seaborn as sn
import librosa
import os
import os.path as osp
import numpy as np
from wav2mel import Wav2Mel
import logging
import sys
import auraloss

# n_fft = 2048
# win_length = None
# hop_length = 512
# new_sample_rate=16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def setup_logger(logger_name, filename):
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        stderr_handler.setLevel(logging.WARNING)
    else:
        stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger

def verifyDir(path):
    if not osp.isdir(path):
        os.mkdir(path)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, cmap='inferno',origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def get_file_data(convertor: Wav2Mel,path, title="", num_frames=3):
    wav, sample_rate = torchaudio.load(path)
    print(wav.shape)
    # wav =wav[:,sample_rate*4:sample_rate*4+sample_rate*num_frames]
    spectogram = convertor.forward(wav)
    print(spectogram.shape)
    plot_spectrogram(spectogram[0], title=title)
    print(title)
    print_stats(wav, sample_rate=sample_rate)


def take_spectograms(convertor:Wav2Mel,path, label, id,spectorams,labels, spectogram_width=1.5, number_of_spectogram=30,out_dir=""):
    wav, sample_rate = torchaudio.load(path)
    # print(wav.shape)
    # print(sample_rate)
    for i in range(2, number_of_spectogram  ):
        start = int(sample_rate * (i * spectogram_width + 1))
        end = int(sample_rate * (i * spectogram_width + 1) + sample_rate * spectogram_width)
        if end>wav.shape[1]:
            continue
        track = wav[:, start:end]
        # track_resampled=T.Resample(sample_rate,new_sample_rate)(track)
        mel=convertor.toMel(track)
        scaled_spectogram=librosa.power_to_db(mel)[:,:128]
        # normalized=(scaled_spectogram-np.mean(scaled_spectogram))/np.std(scaled_spectogram)
        spectorams.append(scaled_spectogram)
        labels.append(label)
        id += 1
    return id


def recover_spectogram(spectogram, n_fft, sample_rate, title=""):
    gf = T.GriffinLim(n_fft)
    new_wav = gf.forward(spectogram)
    torchaudio.save(title + ".wav", new_wav, sample_rate)


# specto=T.Spectrogram(
#     n_fft=n_fft,
#     win_length=win_length,
#     hop_length=hop_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0)
# specto = T.MelSpectrogram(
#     sample_rate=44100,
#     n_fft=n_fft,
#     win_length=win_length,
#     hop_length=hop_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
#     norm='slaney',
#     onesided=True,
#     n_mels=200,
# )


data_dir="data_spectogram"
if __name__ == '__main__':
    logger = setup_logger('__name__', 'spectogram_maker.log')
    verifyDir(data_dir)
    path_Cello = "musicNet/split/Bach_Solo_Cello/train"#0
    path_piano = "musicNet/split/Bach_Solo_Piano/train"#1
    path_violin = "musicNet/split/All_Solo_Violin/train"#2
    path_wind = "musicNet/split/All_Wind_Quintet/train"#3
    # path_string = "musicNet/split/Beethoven_String_Quartet/train"#4
    # path_string = "musicNet\\split\\All_String_Sextet\\train"#1
    labels_dirs=[path_Cello,path_piano,path_violin,path_wind]
    # get_file_data(path_Cello,title="cello")
    id=0
    spectograms=[]
    labels=[]
    last=0
    convertor=Wav2Mel(sample_rate=44100)
    for label,dir in enumerate(labels_dirs):
        print("start "+dir +" at "+str(id) )
        for w in os.listdir(dir):
            path=osp.join(dir,w)
            id=take_spectograms(convertor,path, label,id,spectograms,labels,out_dir=data_dir,number_of_spectogram=300)
            if id-last>1000:
                break
        logger.info("Domain "+dir.split("/")[2]+" is from "+str(last)+" to "+str(id))
        last=id
    print("ends at " + str(id))

    spectograms=np.stack(spectograms)
    labels=np.array(labels)
    np.savez("train_data_wavs",imgs=spectograms,classes=labels)
    # take_spectograms(path_piano, 2)
    # take_spectograms(path_violin, 3)
    # path_wind="musicNet/split/Cambini_Wind_Quintet/train/2075.wav"
    # path_string="musicNet/split/Mozart_String_Quartet/train/1791.wav"
    # get_file_data(path_piano,title="piano")
    # get_file_data(path_violin, title="violin")
    # get_file_data(path_wind, title="wind")
    # get_file_data(path_string,title="string quartet")
