
import torch
import torchaudio
import os
import numpy as np
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import soundfile as sf
import librosa as lib
import scipy.signal as sig
import tensorflow as tf
import seaborn as sn
from model.modules import LatentModel, AmortizedModel,VGGDistance,VGGishDistance,constractiveDistance



class Wav2Mel(torch.nn.Module):
    def __init__(
            self,

            sample_rate: int = 44100,
            resample_rate: int = 16000,
            hop_length: int = 160,
            n_fft: int = 2048,
            f_min: float = 50.0,
            n_mels: int = 128,
            preemph: float = 0.97,
            ref_db: float = 20.0,
            dc_db: float = 100.0,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.f_min = f_min
        self.n_mels = n_mels
        self.preemph = preemph
        self.ref_db = ref_db
        self.dc_db = dc_db
        self.resampler=T.Resample(sample_rate,resample_rate)
        self.wav2sp=T.Spectrogram(
            n_fft=n_fft,
            win_length=None,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            onesided=True,
        )
        self.sp2mel=T.MelScale(
            sample_rate=resample_rate,
            norm='slaney',
            n_mels=self.n_mels,
        )

        # self.melMaker=T.MelSpectrogram(
        #     sample_rate=resample_rate,
        #     n_fft=n_fft,
        #     win_length=None,
        #     hop_length=self.hop_length,
        #     center=True,
        #     pad_mode="reflect",
        #     power=1.0,
        #     norm='slaney',
        #     onesided=True,
        #     n_mels=self.n_mels,
        #         )
        self.mel2Sp = T.InverseMelScale(
            n_stft=(self.n_fft // 2) + 1,
            n_mels=self.n_mels,
            sample_rate=resample_rate,
            norm="slaney"
        )

        self.gf = T.GriffinLim(
            self.n_fft,
            win_length=None,
            hop_length=self.hop_length,
            power=2.0,
            n_iter=82,
        )



    def make_spectogram(self,wav) :
        # resampled=self.resampler.forward(wav)
        resampled=lib.resample(wav,self.sample_rate,self.resample_rate)
        return lib.stft(resampled,n_fft=self.n_fft,hop_length=self.hop_length,win_length=None,center=True)

    # def apply_mel_scale(self, spectogram:torch.Tensor)-> torch.Tensor:
    #     return self.sp2mel.forward((spectogram))

    # def inverse_mel(self,mel:torch.Tensor)-> torch.Tensor:
    #     return self.mel2Sp.forward(mel)

    def toWav(self,mel)-> torch.Tensor:
        # if is_mel:
        #     spectogram=self.mel2Sp.forward(spectogram)
        # return self.gf(spectogram)
        return lib.feature.inverse.mel_to_audio(mel,sr=self.resample_rate,n_fft=self.n_fft,hop_length=self.hop_length,power=2.0,center=True)

    def toMel(self, wav: torch.Tensor) :
        resampled=lib.resample(wav.numpy()[0],self.sample_rate,self.resample_rate)
        return lib.feature.melspectrogram(resampled,sr=self.resample_rate,n_fft=self.n_fft,hop_length=self.hop_length,power=2.0,center=True,n_mels=self.n_mels )
        # return self.melMaker(resampled)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        sp=self.make_spectogram(wav)
        return self.apply_mel_scale(sp)

def plot_spectogram(mel,title=""):
    plt.imshow(mel,cmap="inferno")
    plt.colorbar()
    # sn.heatmap(mel,cmap="inferno")
    # plt.gca().invert_yaxis()
    # plt.axis('off')
    plt.title(title)
    plt.show()

def plot_specs(dir):
    for f in os.listdir(dir):
        sp=np.load(os.path.join(dir,f),allow_pickle=True)["arr_0"]
        plot_spectogram(sp,f)
        print(sp.shape)
        wav=torch.unsqueeze(convertor.toWav(torch.Tensor(sp)[:,:63]),0)
        torchaudio.save(os.path.join("recovred",f.split(".")[0]+".wav"),wav,16000)



if __name__ == '__main__':
    convertor=Wav2Mel(sample_rate=44100)
    # criterion_sound = VGGishDistance([ 4, 5, 7,8,11,13])
    criterion_sound=constractiveDistance()
    # plot_specs("samples_2")
    # data=np.load("train_data_specs_600.npz",allow_pickle=True)
    # samples=[614,2018,2680,3000]
    # samples=[80,616,1405,2820,3150]
    # data=np.load("train_data_mels_256.npz")
    data=np.load("out/cache/preprocess/train_data_mels_4.npz")
    samples=[699,2069,2370,3097]
    spectograms=data["imgs"][samples]

    s2 = [3150,80,616,1405]
    print(data["classes"][samples])
    spectograms_1=data["imgs"][s2]
    s=torch.Tensor(spectograms)
    s2=torch.Tensor(spectograms_1)
    loss=criterion_sound(s[:, None, ...], s2[:, None, ...])
    for i in range(spectograms.shape[0]):
        s=spectograms[i]
        print(s.shape)
        plot_spectogram(s,title=str(samples[i]))
        # wav=torch.unsqueeze(torch.Tensor(convertor.toWav(lib.db_to_power(s))),0)
        # print(wav.shape)
        # torchaudio.save("recovred_lib_spec/"+str(samples[i])+"_lib.wav",wav,16000)
    #
    # samples=["699","2069","2370","3097"]
    # sample_dir="samples_lib"
    # for file in os.listdir(sample_dir):
    #     for suffix in samples:
    #         if file.startswith("105_"+suffix):
    #             mel=np.load(os.path.join(sample_dir,file))["arr_0"]
    #             plot_spectogram(mel,file)
    #             wav=torch.unsqueeze(torch.Tensor(convertor.toWav(lib.db_to_power(mel))),0)
    #             torchaudio.save("recovred_lib/" + file + ".wav", wav, 12000)
    #             print(file)
    # convertor=Wav2Mel(sample_rate=44100)
    # wav,sr=torchaudio.load("C:\\Users\matan\Desktop\School\Master's\Applied DP\project\musicNet\split\Bach_Solo_Piano\\train\\2196.wav")
    # track=wav[:,sr*3:sr*5]
    # torchaudio.save("origin.wav",track,44100)
    # mel=convertor.forward(track)
    # print(mel.shape)
    # recovred=convertor.toWav(mel)
    # torchaudio.save("recovred.wav",recovred,16000)
    # plot_specs("samples")
    # track=T.Resample(44100,16000).forward(track)
    # print(track.shape)
    # torchaudio.save("resampled.wav",track,16000)
    # melspectrogram = MelSpectrogram(
    # sample_rate=16000,
    # n_fft=n_fft,
    # win_length=win_length,
    # hop_length=hop_length,
    # center=True,
    # pad_mode="reflect",
    # power=2.0,
    # norm='slaney',
    # onesided=True,
    # n_mels=n_mels,
    # )
    # mel=melspectrogram.forward(track)
    # # melMaker=Wav2Mel()
    # # mel_spectogram=melMaker.forward(track,sr)
    # plt.imshow(T.AmplitudeToDB().forward(mel)[0])
    # plt.show()
    # print(mel.shape)
    # inv_mel=T.InverseMelScale(
    #     n_mels=n_mels,
    #     n_stft=(n_fft//2)+1,
    #     sample_rate=16000,
    #     norm="slaney",
    # )
    # sp=inv_mel.forward(mel)
    # print(sp.shape)
    # gl=T.GriffinLim(
    #     n_fft=n_fft,
    #     win_length=win_length,
    #     hop_length=hop_length,
    #     power=2.0,
    #     length=32000,
    #     n_iter=82
    # )
    # new_wav=gl.forward(sp)
    # print(new_wav.shape)
    #
    # # print(mel_spectogram.shape)
    # # melRecover=Mel2Wav()
    # # recovred=melRecover.forward(mel_spectogram)
    # # vocoder=Vocoder.from_pretrained("https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt").cuda()
    # # with torch.no_grad:
    # #     wav,sr=vocoder.generate([mel])
    # #     torchaudio.save("recovred.wav", new_wav, 16000)
    #
    # torchaudio.sox_effects.shutdown_sox_effects()
