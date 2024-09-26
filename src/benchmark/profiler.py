import sys
import os
import torch
import time
import torchaudio
import json
import pickle
import statistics
from torchvision.datasets import DatasetFolder
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dodanie ścieżki nadrzędnej do sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.sound_features import fft_cache, dominant_freq, fundamental_freq, stft_spectrogram
from features.sound_features import mel_energy, bark_energy, cqt_energy, erb_energy, gammatone_energy
from features.sound_features import lpc_features, mfcc, bfcc, gfcc, rms, zcr, teo, psd, asd
from features.sound_features import spectral_entropy, spectral_centroid, spectral_flux, spectral_contrast

class FeaturesExtractorProfiler:
    def __init__(self, frame_size_ms: int, hop_length_ms: int, features: dict[str, any] = None) -> None:
        self.frame_size_ms = frame_size_ms
        self.hop_length_ms = hop_length_ms
        self.features = features
        self.stats = {name: {'times': [], 'sizes': []} for name in features.keys()}

    def stream_audio(self, filepath: str, loader):
        waveform, sample_rate = loader(filepath)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        frame_size = int(self.frame_size_ms * sample_rate / 1000)
        hop_length = int(self.hop_length_ms * sample_rate / 1000)

        waveform = waveform / torch.max(torch.abs(waveform))

        num_frames = (waveform.size(0) - frame_size) // hop_length + 1

        for i in range(num_frames):
            start_idx = i * hop_length
            frame = waveform[start_idx:start_idx + frame_size]

            if frame.size(0) < frame_size:
                break

            for name, feature_fn in self.features.items():
                start_time = time.time()

                feature_value = feature_fn(frame, sample_rate).reshape(-1)

                end_time = time.time()
                processing_time = end_time - start_time
                data_size = feature_value.numel()

                self.stats[name]['times'].append(processing_time)
                self.stats[name]['sizes'].append(data_size)

        fft_cache.reset()

    def save_frame_stats(self, filepath: str) -> None:
        """
        Zapisuje statystyki ramki do pliku pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_frame_stats(self, filepath: str) -> None:
        """
        Wczytuje statystyki ramki z pliku pickle.
        """
        with open(filepath, 'rb') as f:
            self.stats = pickle.load(f)

class AudioDatasetProfiler(DatasetFolder):
    def __init__(self, dir: str, features_extractor):
        super().__init__(
            root=dir,
            loader=self.audio_loader,
            extensions=('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        )
        self.feature_extractor = features_extractor

    def audio_loader(self, path: str):
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        self.feature_extractor.stream_audio(path, self.audio_loader)
        return None

def dict_to_formatted_str(d, var_name):
    formatted_str = f"{var_name} = " + json.dumps(d, indent=4) + "\n"
    return formatted_str

def main():
    # Definicja funkcji
    features = {
        'dominant_freq': dominant_freq,
        'fundamental_freq': fundamental_freq,
        'stft': stft_spectrogram,
        'mel_energy': mel_energy,
        'bark_energy': bark_energy,
        'cqt_energy': cqt_energy,
        'erb_energy': erb_energy,
        'gammatone_energy': gammatone_energy,
        'lpc': lpc_features,
        'mfcc': mfcc,
        'bfcc': bfcc,
        'gfcc': gfcc,
        'rms': rms,
        'zcr': zcr,
        'teo': teo,
        'psd': psd,
        'asd': asd,
        'spectral_entropy': spectral_entropy,
        'spectral_centroid': spectral_centroid,
        'spectral_flux': spectral_flux,
        'spectral_contrast': spectral_contrast,
    }

    # Ścieżka do pliku statystyk ramki
    frame_stats_file = 'frame_stats.pkl'

    if os.path.exists(frame_stats_file):
        print("Wczytywanie zapisanych statystyk ramki...")
        features_extractor_profiler = FeaturesExtractorProfiler(
            frame_size_ms=40,
            hop_length_ms=40,
            features=features
        )
        features_extractor_profiler.load_frame_stats(frame_stats_file)
    else:
        print("Przetwarzanie danych i zapisywanie statystyk ramki...")
        # Inicjalizacja profilerów
        features_extractor_profiler = FeaturesExtractorProfiler(
            frame_size_ms=40,
            hop_length_ms=40,
            features=features
        )
        audio_dataset_profiler = AudioDatasetProfiler(
            dir='../data/output',
            features_extractor=features_extractor_profiler
        )

        # Przetwarzanie danych
        for _ in audio_dataset_profiler:
            pass

        # Zapisanie statystyk ramki do pliku pickle
        features_extractor_profiler.save_frame_stats(frame_stats_file)

    # Analiza statystyk
    processing_times = {}
    for feature_name, data in features_extractor_profiler.stats.items():
        times = data['times']
        if len(times) > 1:
            std_time_ms = statistics.stdev(times) * 1000  # Sekundy na milisekundy
        else:
            std_time_ms = 0.0

        processing_times[feature_name] = {
            'min_time_ms': min(times) * 1000,       # Konwersja na milisekundy
            'avg_time_ms': (sum(times) / len(times)) * 1000,
            'max_time_ms': max(times) * 1000,
            'std_time_ms': std_time_ms
        }

    data_sizes = {}
    for feature_name, data in features_extractor_profiler.stats.items():
        sizes = data['sizes']

        data_sizes[feature_name] = {
            'size': min(sizes)
        }

    # Konwersja słowników na sformatowane ciągi znaków
    processing_times_str = dict_to_formatted_str(processing_times, "processing_times")
    data_sizes_str = dict_to_formatted_str(data_sizes, "data_sizes")

    # Zapisanie do plików
    with open('processing_times.py', 'w') as f:
        f.write("# Processing Times\n")
        f.write(processing_times_str)

    with open('data_sizes.py', 'w') as f:
        f.write("# Data Sizes\n")
        f.write(data_sizes_str)

    print("Statystyki zostały zapisane do plików.")

if __name__ == "__main__":
    main()
