import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler
from features.sound_features import *
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

# Import your dataset and feature extractor
from dataset.audio_dataset import AudioDataset
from dataset.features_extractor import FeaturesExtractor
from features.sound_features import *

def remove_nan_and_inf(X):
	"""Replace NaN and inf with zero."""
	X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
	return X

def generate_latex_table(pca_loadings, feature_names):
	"""
	Generate LaTeX table for PCA component loadings.
	"""
	num_components = pca_loadings.shape[1]
	header = " & " + " & ".join(feature_names) + " \\\\ \\hline\n"
	rows = []
	
	for i in range(num_components):
		row = f"Komponent {i+1}"
		for j in range(len(feature_names)):
			row += f" & {int(pca_loadings[j, i])}"
		row += " \\\\ \\hline"
		rows.append(row)
	
	table = "\\begin{tabular}{|c|" + "c|" * len(feature_names) + "}\n\\hline\n"
	table += header
	table += "\n".join(rows)
	table += "\n\\end{tabular}"
	
	return table

def plot_pca_loadings(pca_loadings, feature_names):
    """
    Generate a plot for PCA component loadings.
    """
    num_components = min(3, pca_loadings.shape[1])  # We limit it to the first 3 components
    x = np.arange(len(feature_names))  # X-axis represents feature indices (cechy)

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Define different colors for each component
    colors = ['r', 'g', 'b']
    
    for i in range(num_components):
        # Plot each component with a different color and markers
        plt.plot(x, pca_loadings[:len(feature_names), i], marker='o', color=colors[i], label=f'Component {i+1}', linestyle='-', linewidth=2)
    
    # Add labels and title
    plt.xticks(x, feature_names, rotation=45, ha='right')  # Feature names on the x-axis
    plt.xlabel('Sound features')
    plt.ylabel('PCA loadings')
    plt.title('PCA loadings for the first 3 components')

    # Add grid for clarity
    plt.grid(True)

    # Add legend
    plt.legend()

    # Save plot to file
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig('pca_loadings_plot_norm.png')  # Save as PNG file


def evaluate_combined_features_pca(data_loader, feature_names):
	all_frames = []
	all_pads = []

	# Extract features and PAD values from the dataset
	for batch in data_loader:
		features = batch['features']
		pad = batch['pad']  # 'pad' is a list of floats, e.g., [0.2, 0.8, 0.6]
		pad = np.array(pad, dtype=float).reshape(1, 3)  # Ensure pad is a numpy array of shape (1, 3)

		feature_vectors = []
		# Extract feature vectors for all given feature names
		for feature_name in feature_names:
			feature_vector = features[feature_name]['vector'].numpy()
			frame_size = features[feature_name]['size']
			num_frames = feature_vector.size // frame_size
			reshaped_features = feature_vector.reshape(num_frames, frame_size)
			feature_vectors.append(reshaped_features)

		# Concatenate feature vectors for each frame
		combined_features = np.hstack(feature_vectors)  # Combine all features into a single array
		all_frames.append(combined_features)

		# Repeat PAD values for each frame in the sample
		pad_repeated = np.tile(pad, (num_frames, 1))  # Shape: (num_frames, 3)
		all_pads.append(pad_repeated)

	# Concatenate all frames and PAD values from all audio samples
	X = np.vstack(all_frames)  # Shape: [total_frames, total_features]
	X = remove_nan_and_inf(X)
	y_pads = np.vstack(all_pads)  # Shape: [total_frames, 3]

	# Apply standardization (normalization) to the features
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# Perform PCA on the scaled data
	pca = PCA()
	X_pca = pca.fit_transform(X_scaled)
	explained_variance_ratio = pca.explained_variance_ratio_

	# Wydobywanie ładunków PCA
	pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

	# Display feature importance for each component
	print("\nIstotność cech w pierwszych 3 komponentach PCA:")
	for i in range(3):  # Limit to first 3 components
		print(f"\nKomponent PCA {i+1}:")
		for j, feature_name in enumerate(feature_names):
			print(f"  Cecha: {feature_name}, Ładunek: {pca_loadings[j, i]:.4f}")

	#latex_table = generate_latex_table(pca_loadings[:, :3], feature_names)  # Limit to first 3 components
	#print(latex_table)
	plot_pca_loadings(pca_loadings, feature_names)

	# Continue with cumulative variance and correlation analysis...
	pad_labels = ['Pleasure (P)', 'Arousal (A)', 'Dominance (D)']
	max_components = min(3, len(explained_variance_ratio))  # Ensure we don't exceed the number of components

	print(f"\nAnaliza PCA dla zestawu cech: {', '.join(feature_names)}")

	cumulative_variance = 0.0
	cumulative_scores = np.zeros(X_pca.shape[0])

	for n_components in range(1, max_components + 1):
		# Update cumulative variance
		cumulative_variance += explained_variance_ratio[n_components - 1]
		cumulative_variance_percent = cumulative_variance * 100

		# Sum the first n_components PCA scores for each sample
		cumulative_scores += X_pca[:, n_components - 1]

		print(f"\nSumaryczne wyniki dla pierwszych {n_components} komponentów PCA:")
		print(f"  Skumulowana wyjaśniona wariancja: {cumulative_variance_percent:.2f}%")

		# Oblicz korelacje między sumarycznym wynikiem PCA a wartościami PAD
		for j in range(3):  # For each PAD dimension
			y = y_pads[:, j]
			x = cumulative_scores

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
				print(f"    Błąd w obliczaniu korelacji: {e}")
	print("-" * 50)

def evaluate_all_features_pca(data_loader, transformers):
	# Define a list of feature names you want to combine for PCA
	combined_feature_names = list(transformers.keys())
	print(f"\nEvaluating combined features: {combined_feature_names}")
	evaluate_combined_features_pca(data_loader, combined_feature_names)

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

	# Run the PCA evaluation for all features combined
	evaluate_all_features_pca(data_loader, transformers)
