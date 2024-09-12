import os
import torch

class FeaturesExtractor():
	def __init__(self, frame_size_ms: int, hop_length_ms: int, features: dict[str, any] = None, precomputed_dir=None) -> None:
		self.frame_size_ms = frame_size_ms
		self.hop_length_ms = hop_length_ms
		self.features = features

		if precomputed_dir:
			os.makedirs(precomputed_dir, exist_ok=True)
			self.precomputed_dir = precomputed_dir

	def stream_audio(self, filepath: str, loader):
		# Extract the filename for use in caching
		filename = os.path.splitext(os.path.basename(filepath))[0]

		# Check if precomputed directory is provided and caching is enabled
		if self.precomputed_dir:
			cached_features = {}
			all_features_cached = True

			for feature_name in self.features.keys():
				feature_filename = os.path.join(self.precomputed_dir, f"{feature_name}_{filename}.pt")
				
				if os.path.exists(feature_filename):
					cached_features[feature_name] = {
						'vector': torch.load(feature_filename),
						'size': torch.load(feature_filename).shape[1]
					}
				else:
					all_features_cached = False
					break  # If any feature is missing, stop and compute all features

			if all_features_cached:
				return cached_features

		waveform, sample_rate = loader(filepath)
		
		if waveform.shape[0] > 1:
			waveform = torch.mean(waveform, dim=0)

		frame_size = int(self.frame_size_ms * sample_rate / 1000)
		hop_length = int(self.hop_length_ms * sample_rate / 1000)

		waveform = waveform / torch.max(torch.abs(waveform))

		features_info = {}
		num_frames = (waveform.size(0) - frame_size) // hop_length + 1

		# Precompute the feature size using the first frame
		first_frame = waveform[0:frame_size]
		for name, feature_fn in self.features.items():
			feature_value = feature_fn(first_frame, sample_rate)
			feature_size = feature_value.reshape(-1).shape[0]

			features_info[name] = {
				'vector': torch.empty((num_frames, feature_size)),
				'size': feature_size
			}

		# Compute features for each frame
		for i in range(num_frames):
			start_idx = i * hop_length
			frame = waveform[start_idx:start_idx + frame_size]

			if frame.size(0) < frame_size:
				break

			for name, feature_fn in self.features.items():
				feature_value = feature_fn(frame, sample_rate)
				features_info[name]['vector'][i] = feature_value.reshape(-1)

		# Save computed features to disk if caching is enabled
		if self.precomputed_dir:
			for feature_name, feature_info in features_info.items():
				feature_filename = os.path.join(self.precomputed_dir, f"{feature_name}_{filename}.pt")
				torch.save(feature_info['vector'], feature_filename)

		return features_info
