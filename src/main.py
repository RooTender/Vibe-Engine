import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.audio_dataset import AudioDataset
from dataset.features_extractor import FeaturesExtractor
from features.sound_features import *
from networks.rnn import MultiFeatureRNNModel

features = {
	'dominant_freq': dominant_freq,
	'fundamental_freq': fundamental_freq,

	'stft': stft_spectrogram,

	#'mel_energy': mel_energy,
	#'bark_energy': bark_energy,
	#'cqt_energy': cqt_energy,
	#'erb_energy': erb_energy,
	#'gammatone_energy': gammatone_energy,
#
	#'lpc': lpc_features,
#
	#'mfcc': mfcc,
	#'bfcc': bfcc,
	#'gfcc': gfcc,
#
	#'rms': rms,
	#'zcr': zcr,
	#'teo': teo,
#
	#'psd': psd,
	#'asd': asd,
	#
	#'spectral_entropy': spectral_entropy,
	#'spectral_centroid': spectral_centroid,
	#'spectral_flux': spectral_flux,
	#'spectral_contrast': spectral_contrast,
}

features_extractor = FeaturesExtractor(frame_size_ms=40, hop_length_ms=40, features=features, precomputed_dir='../data/precomputed25')
audio_dataset = AudioDataset(dir='../data/output', features_extractor=features_extractor)

data_loader = DataLoader(audio_dataset, batch_size=1)

hidden_size = 64    # Rozmiar ukrytego stanu
output_size = 3     # Rozmiar wyjścia (np. dla wektora PAD)
num_layers = 2      # Liczba warstw rekurencyjnych
dropout = 0.5       # Współczynnik dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicjalizacja zmiennych
model_initialized = False
feature_sizes = None

num_epochs = 20

criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for batch in data_loader:
        labels_batch = batch['label']
        features_info = batch['features']
        pad_values = torch.tensor(batch['pad'], dtype=torch.float32).to(device)

        feature_tensors = []
        current_feature_sizes = []
        for feature_name, feature_data in features_info.items():
            vector = feature_data['vector'].to(device).flatten()
            size = feature_data['size']
            current_feature_sizes.append(size)

            # Obliczenie liczby ramek
            num_frames = vector.numel() // size

            # Przygotowanie tensora o kształcie (batch_size, seq_length, feature_size)
            x_feature = vector.view(num_frames, size)
            x_feature = x_feature.unsqueeze(0)  # Dodanie wymiaru batch_size
            feature_tensors.append(x_feature)

        if not feature_tensors:
            continue  # Pomiń tę próbkę, jeśli nie ma żadnych cech

        # Inicjalizacja modelu po pobraniu rozmiarów cech
        if not model_initialized:
            feature_sizes = current_feature_sizes
            model = MultiFeatureRNNModel(feature_sizes=feature_sizes, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model_initialized = True

        # Forward pass
        outputs = model(feature_tensors)

        # Obliczenie straty
        loss = criterion(outputs, pad_values)

        # Backward pass i optymalizacja
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
