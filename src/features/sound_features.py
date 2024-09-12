import torch
import torchaudio
import fcwt
from .superlet import superlets
import numpy as np

def stft_spectrogram(frame, _):
	return torchaudio.transforms.Spectrogram()(frame)

def stft_spectrogram_9(frame, _):
	return torchaudio.transforms.Spectrogram(power=9)(frame)

def cwt_spectrogram(frame, sample_rate):
	_, result = fcwt.cwt(
		input=frame.numpy(), 
		fs=sample_rate, f0=1, f1=sample_rate // 2, fn=50, 
		nthreads=8
	)

	return torch.tensor(np.abs(np.stack(result, axis=0)))

def slt_spectrogram(frame, sample_rate):
	result = superlets(data=frame,
				  fs=sample_rate,
				  foi=np.linspace(1, sample_rate // 2, 50),
				  c1=2, ord=(3, 5))
	
	return torch.tensor(np.abs(np.stack(result, axis=0)))


def mfcc(frame, sample_rate):
	return torchaudio.transforms.MFCC(
		sample_rate=sample_rate,
		n_mfcc=13,  # Typically 13 MFCCs are used
		melkwargs={
			"n_mels": 40,         # Keep 40 mel bands
			"n_fft": 640,         # FFT size large enough for 40ms window (640 samples)
			"hop_length": 640,    # 40ms hop length (640 samples)
			"win_length": 640     # 40ms window length (640 samples)
		}
	)(frame)
