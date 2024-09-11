import torchaudio
import fcwt
from .superlet import superlets
import numpy as np

def stft_spectrogram(frame, _):
	return torchaudio.transforms.Spectrogram()(frame)

def stft_spectrogram_9(frame, _):
	return torchaudio.transforms.Spectrogram(power=9)(frame)

def cwt_spectrogram(frame, sample_rate):
	_, output = fcwt.cwt(
		input=frame.numpy(), 
		fs=sample_rate, f0=1, f1=sample_rate // 2, fn=400, 
		nthreads=8)
	return output

def slt_spectrogram(frame, sample_rate):
	return superlets(data=frame,
				  fs=sample_rate,
				  foi=np.linspace(1, sample_rate // 2, 100),
				  c1=2, ord=(5, 10))

def mfcc(frame, sample_rate):
	return torchaudio.transforms.MFCC(
		sample_rate=sample_rate,
		n_mfcc=13,
		melkwargs={"n_mels": 64}
	)(frame)