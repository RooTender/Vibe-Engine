from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# Import your dataset and feature extractor
from dataset.audio_dataset import AudioDataset
from dataset.features_extractor import FeaturesExtractor
from features.sound_features import *

def remove_nan_and_inf(X):
    """Replace NaN and inf with zero."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def calculate_class_priors(y_labels):
    """Calculate class priors based on the frequency of each class."""
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    total_count = len(y_labels)
    priors = counts / total_count
    inverse_priors = 1 / priors  # Inverse proportionality
    normalized_priors = inverse_priors / inverse_priors.sum()  # Normalize to sum to 1
    return normalized_priors

def evaluate_feature_lda(data_loader, feature_name):
    X = []
    y_labels = []

    # Collect all data
    for batch in data_loader:
        features = batch['features']
        label = batch['label']  # Get emotion label instead of PAD

        feature_vector = features[feature_name]['vector'].numpy().astype(np.float32)
        frame_size = features[feature_name]['size']
        num_frames = feature_vector.size // frame_size
        reshaped_features = feature_vector.reshape(num_frames, frame_size)

        label_repeated = [label] * num_frames  # Repeat the emotion label for each frame
        y_labels.extend(label_repeated)
        X.append(reshaped_features)

    # Concatenate all batches
    X = np.vstack(X)
    y_labels = np.array(y_labels).ravel()  # Ensure y_labels is 1D

    # Remove NaN and Inf
    X = remove_nan_and_inf(X)

    # Encode emotion labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    # Standardize the data (mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate class priors based on label frequencies
    priors = calculate_class_priors(y_encoded)

    print(f"\nLDA Analysis for {feature_name} with balanced class priors:")

    # LDA analysis for emotion labels with manually set priors
    lda = LinearDiscriminantAnalysis(priors=priors)
    if len(np.unique(y_encoded)) < 2:
        print(f"Not enough classes to perform LDA.")
        return

    try:
        # Fit the LDA model
        lda.fit(X_scaled, y_encoded)
        y_pred = lda.predict(X_scaled)

        # Classification report
        report = classification_report(y_encoded, y_pred, target_names=label_encoder.classes_, zero_division=1)

        print(f"Classification Report:")
        print(report)
    except Exception as e:
        print(f"Error during LDA: {e}")

    print("-" * 50)


def evaluate_all_features_lda(data_loader, transformers):
    for feature_name in transformers.keys():
        print(f"\nEvaluating feature: {feature_name}")
        evaluate_feature_lda(data_loader, feature_name)



if __name__ == "__main__":
	# Define features and feature extractor
	transformers = {
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

	# Set up feature extractor and dataset
	features_extractor = FeaturesExtractor(
		frame_size_ms=40,
		hop_length_ms=40,
		features=transformers,
		precomputed_dir='../data/precomputed'
	)
	audio_dataset = AudioDataset(
		dir='../data/output',
		features_extractor=features_extractor
	)
	data_loader = DataLoader(audio_dataset, batch_size=1)

	# Run the LDA evaluation for all features
	evaluate_all_features_lda(data_loader, transformers)
