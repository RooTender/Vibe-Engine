import torch

class FeaturesExtractor():
	def __init__(self, frame_size_ms: int, hop_length_ms: int, features: dict[str, any] = None) -> None:
		self.frame_size_ms = frame_size_ms
		self.hop_length_ms = hop_length_ms
		self.features = features

	def stream_audio(self, waveform, sample_rate):
		frame_size = int(self.frame_size_ms * sample_rate / 1000)
		hop_length = int(self.hop_length_ms * sample_rate / 1000)

		waveform = waveform / torch.max(torch.abs(waveform))

		features_info = {}
		num_frames = (waveform.size(1) - frame_size) // hop_length + 1

		# Deducing sizes for preprocessing
		first_frame = waveform[:, 0:frame_size]
		for name, feature_fn in self.features.items():
			feature_value = feature_fn(first_frame)
			feature_size = feature_value.reshape(-1).shape[0]

			features_info[name] = {
				'vector': torch.empty((num_frames, feature_size)),
				'size': feature_size
			}

		# Actual computing
		for i in range(num_frames):
			start_idx = i * hop_length
			frame = waveform[:, start_idx:start_idx + frame_size]

			if frame.size(1) < frame_size:
				break

			for name, feature_fn in self.features.items():
				feature_value = feature_fn(frame)
				features_info[name]['vector'][i] = feature_value.reshape(-1)

		return features_info
