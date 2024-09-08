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

		waveform_features = {}
		for name in self.features.keys():
			waveform_features[name] = []

		for i in range(0, waveform.size(1), hop_length):
			frame = waveform[:, i:i+frame_size]

			if frame.size(1) < frame_size:
				break

			features = self.extract_features(frame)

			for name, feature_value in features.items():
				waveform_features[name].append(feature_value)
		
		for name in waveform_features:
			waveform_features[name] = torch.stack(waveform_features[name])

		return waveform_features

	def extract_features(self, frame: any):
		result = {}

		for name, feature_fn in self.features.items():
			result[name] = feature_fn(frame)

		return result
