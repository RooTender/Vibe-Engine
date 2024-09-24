import torch
import torchaudio
import fcwt
from .superlet import superlets
from .erb_banks import EquivalentRectangularBandwidth
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies
from spafe.frequencies.fundamental_frequencies import compute_yin
from spafe.fbanks.mel_fbanks import mel_filter_banks
from spafe.fbanks.bark_fbanks import bark_filter_banks
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
from spafe.features.lpc import lpc
from spafe.features.mfcc import mfcc as mfcc_
from spafe.features.bfcc import bfcc as bfcc_
from spafe.features.gfcc import gfcc as gfcc_
import librosa
from scipy import signal, fft
import numpy as np

import os, sys

win_length = 25

class SurpressWarnings:
	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout


def dominant_freq(frame, sample_rate):
	dominant_freqs = get_dominant_frequencies(frame, fs=sample_rate, nfft=512)

	if len(dominant_freqs) < 2:
		dominant_freqs = np.pad(dominant_freqs, (0, 2 - len(dominant_freqs)), mode='constant')
		
	return torch.tensor(dominant_freqs)

def fundamental_freq(frame, sample_rate):
	if np.max(np.abs(frame.numpy())) == 0:
		return torch.zeros(4)

	win_len = win_length / 1000

	pitch, harmonic_rates, argmins, times = compute_yin(frame, fs=sample_rate, win_len=win_len)
	result = np.concatenate([pitch, harmonic_rates, argmins, times], axis=None)

	if result.shape[0] == 4:
		return torch.tensor(result)
	else:
		return torch.zeros(4)


def stft_spectrogram(frame, _):
	return torchaudio.transforms.Spectrogram()(frame)

def stft_spectrogram_3(frame, _):
	return torchaudio.transforms.Spectrogram(power=3)(frame)

def stft_spectrogram_6(frame, _):
	return torchaudio.transforms.Spectrogram(power=6)(frame)

def stft_spectrogram_9(frame, _):
	return torchaudio.transforms.Spectrogram(power=9)(frame)

def stft_spectrogram_12(frame, _):
	return torchaudio.transforms.Spectrogram(power=12)(frame)

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


class FFTCache:
	"""Cache for storing FFT results for frames."""
	def __init__(self):
		self.cache = {}

	def get_fft(self, frame, nfft=512):
		"""Return the FFT of the frame from cache or compute if not cached."""
		if isinstance(frame, torch.Tensor):
			frame = frame.numpy()

		frame_key = frame.tobytes()  # Convert the frame to bytes to use as a dictionary key

		if frame_key not in self.cache:
			# Compute FFT and store it in the cache
			frame_fft = np.fft.fft(frame, nfft)[:nfft // 2 + 1]
			self.cache[frame_key] = frame_fft

		return self.cache[frame_key]
	
	def reset(self):
		"""Clear the cache."""
		self.cache.clear()


# Initialize the global FFT cache
fft_cache = FFTCache()

# Updated functions using the cache
def mel_energy(frame, sample_rate):
	frame_fft = fft_cache.get_fft(frame)
	mel_fbanks, _ = mel_filter_banks(fs=sample_rate)
	return torch.tensor(np.dot(mel_fbanks, np.abs(frame_fft) ** 2))

def bark_energy(frame, sample_rate):
	frame_fft = fft_cache.get_fft(frame)
	bark_fbanks, _ = bark_filter_banks(fs=sample_rate)
	return torch.tensor(np.dot(bark_fbanks, np.abs(frame_fft) ** 2))

def gammatone_energy(frame, sample_rate):
	frame_fft = fft_cache.get_fft(frame)
	gammatone_fbanks, _ = gammatone_filter_banks(fs=sample_rate)
	return torch.tensor(np.dot(gammatone_fbanks, np.abs(frame_fft) ** 2))

def cqt_energy(frame, sample_rate):
	frame = frame.numpy()  # Convert to numpy array for librosa compatibility
	cqt = np.abs(librosa.cqt(frame, sr=sample_rate))
	return torch.tensor(np.sum(cqt ** 2, axis=0))

def erb_energy(frame, sample_rate):
	frame_fft = fft_cache.get_fft(frame)
	erb_bank = EquivalentRectangularBandwidth(len(frame_fft), sample_rate, 40, 20, sample_rate / 2)
	erb_filters = erb_bank.filters
	erb_amplitudes = np.dot(np.abs(frame_fft) ** 2, erb_filters)
	return torch.tensor(erb_amplitudes)


def lpc_features(frame, sample_rate):
	order=13

	frame = frame.numpy()
	if np.max(np.abs(frame)) == 0:
		return torch.zeros(order * 2 + 2)
	
	try:
		lpc_coeffs, lpc_errors = lpc(frame, sample_rate, order=order)

		lpc_coeffs = np.array(lpc_coeffs)
		lpc_errors = np.array([lpc_errors])

		lpc_result = np.concatenate([lpc_coeffs, lpc_errors], axis=None)

		return torch.tensor(lpc_result)
	except:
		return torch.zeros(order * 2 + 2)


def mfcc(frame, sample_rate):
	return torch.tensor(mfcc_(frame, sample_rate))

def bfcc(frame, sample_rate):
	return torch.tensor(bfcc_(frame, sample_rate))

def gfcc(frame, sample_rate):
	return torch.tensor(gfcc_(frame, sample_rate))


def rms(frame, _):
	return torch.tensor(librosa.feature.rms(y=frame.numpy()))

def zcr(frame, _):
	frame = frame.numpy()
	return torch.tensor(librosa.feature.zero_crossing_rate(y=frame, frame_length=len(frame)))

def autocorrelation(frame, _):
	return torch.tensor(librosa.autocorrelate(y=frame.numpy()))

def teo(frame, _):
	signal = np.asarray(frame)
	result = np.zeros_like(frame)
	
	result[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
	
	return torch.tensor(result)


def psd(frame, sample_rate):
	frame = frame.numpy()
	_, result = signal.welch(frame, fs=sample_rate, nperseg=len(frame))
	return torch.tensor(result)

def asd(frame, sample_rate):
	return torch.sqrt(psd(frame, sample_rate))

def spectral_entropy(frame, sample_rate):
	psd_value = psd(frame, sample_rate).numpy()
	
	total_power = np.sum(psd_value)
	if total_power == 0:
		return torch.zeros(1)
	
	psd_value = psd_value / total_power
	
	epsilon = 1e-12  # avoid log(0)
	spectral_entropy = -np.sum(psd_value * np.log2(psd_value + epsilon))
	
	return torch.tensor(spectral_entropy)

def spectral_centroid(frame, sample_rate):
	return torch.tensor(librosa.feature.spectral_centroid(y=frame.numpy(), sr=sample_rate, n_fft=512))

previous_spectrum = None
def spectral_flux(frame, _):
	global previous_spectrum
	spectrum = np.abs(fft.fft(frame))

	if previous_spectrum is None:
		previous_spectrum = spectrum
		return torch.zeros(1)

	flux = np.sum((spectrum - previous_spectrum) ** 2)
	previous_spectrum = spectrum

	return torch.tensor(flux)

def spectral_contrast(frame, sample_rate):
	return torch.tensor(librosa.feature.spectral_contrast(y=frame.numpy(), sr=sample_rate, n_fft=512))
