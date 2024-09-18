import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Import your dataset and feature extractor
from dataset.audio_dataset import AudioDataset
from dataset.features_extractor import FeaturesExtractor
from features.sound_features import *

def remove_nan_and_inf(X):
	"""Replace NaN and inf with zero."""
	X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
	return X

import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy.stats import pearsonr, spearmanr

def evaluate_feature_pca(data_loader, feature_name):
	# Initialize Incremental PCA
	n_components = 3
	ipca = IncrementalPCA(n_components=n_components)
	
	pad_labels = ['Pleasure (P)', 'Arousal (A)', 'Dominance (D)']
	cumulative_scores = None
	y_pads_list = []
	
	# First pass: Fit the IPCA model in batches
	for batch in data_loader:
		features = batch['features']
		pad = batch['pad']
		pad = np.array(pad, dtype=np.float32).reshape(1, 3)

		feature_vector = features[feature_name]['vector'].numpy().astype(np.float32)
		frame_size = features[feature_name]['size']
		num_frames = feature_vector.size // frame_size
		reshaped_features = feature_vector.reshape(num_frames, frame_size)

		pad_repeated = np.tile(pad, (num_frames, 1))
		y_pads_list.append(pad_repeated)

		X_batch = remove_nan_and_inf(reshaped_features)

		# Fit the IPCA model on the current batch
		ipca.partial_fit(X_batch)

	# Initialize variables for cumulative scores and PAD values
	cumulative_scores = np.zeros((0, n_components), dtype=np.float32)
	y_pads = np.vstack(y_pads_list)

	# Second pass: Transform data and compute cumulative scores
	for batch in data_loader:
		features = batch['features']

		feature_vector = features[feature_name]['vector'].numpy().astype(np.float32)
		frame_size = features[feature_name]['size']
		num_frames = feature_vector.size // frame_size
		reshaped_features = feature_vector.reshape(num_frames, frame_size)

		X_batch = remove_nan_and_inf(reshaped_features)

		# Transform the batch using the fitted IPCA model
		X_pca_batch = ipca.transform(X_batch)
		cumulative_scores = np.vstack((cumulative_scores, X_pca_batch))

	explained_variance_ratio = ipca.explained_variance_ratio_

	print(f"\nAnaliza PCA dla {feature_name}:")

	cumulative_variance = 0.0
	total_cumulative_scores = np.zeros(cumulative_scores.shape[0], dtype=np.float32)

	for n_comp in range(1, n_components + 1):
		# Update cumulative variance
		cumulative_variance += explained_variance_ratio[n_comp - 1]
		cumulative_variance_percent = cumulative_variance * 100

		# Sum the first n_comp PCA scores for each sample
		total_cumulative_scores += cumulative_scores[:, n_comp - 1]

		print(f"\nSumaryczne wyniki dla pierwszych {n_comp} komponentów PCA:")
		print(f"  Skumulowana wyjaśniona wariancja: {cumulative_variance_percent:.2f}%")

		# Compute correlations between cumulative PCA scores and PAD values
		for j in range(3):  # For each PAD dimension
			y = y_pads[:, j]
			x = total_cumulative_scores

			# Check if inputs are valid
			if x.size < 2 or y.size < 2:
				print(f"  {pad_labels[j]}:")
				print("    Not enough data points to compute correlation.")
				continue
			if np.all(x == x[0]):
				print(f"  {pad_labels[j]}:")
				print("    Cannot compute correlation: PCA scores are constant.")
				continue
			if np.all(y == y[0]):
				print(f"  {pad_labels[j]}:")
				print("    Cannot compute correlation: PAD value is constant.")
				continue
			try:
				pearson_corr, pearson_p = pearsonr(x, y)
				spearman_corr, spearman_p = spearmanr(x, y)
				if np.isnan(pearson_corr) or np.isnan(spearman_corr):
					print(f"  {pad_labels[j]}:")
					print("    Correlation coefficient is NaN.")
					continue
				print(f"  {pad_labels[j]}:")
				print(f"    Korelacja Pearsona: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
				print(f"    Korelacja Spearmana: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
			except Exception as e:
				print(f"  {pad_labels[j]}:")
				print(f"    Error computing correlation: {e}")
	print("-" * 50)


def evaluate_all_features_pca(data_loader, transformers):
	for feature_name in transformers.keys():
		print(f"\nEvaluating feature: {feature_name}")
		evaluate_feature_pca(data_loader, feature_name)

if __name__ == "__main__":
	# Define features and feature extractor
	transformers = {
		'STFT': stft_spectrogram,
		#'STFT with Power 3': stft_spectrogram_3,
		#'STFT with Power 6': stft_spectrogram_6,
		#'STFT with Power 9': stft_spectrogram_9,
		#'STFT with Power 12': stft_spectrogram_12,
		#'CWT': cwt_spectrogram,
		#'SLT': slt_spectrogram,
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

	# Run the PCA evaluation for all features
	evaluate_all_features_pca(data_loader, transformers)
