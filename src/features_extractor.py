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

		for i in range(0, waveform.size(1), hop_length):
			frame = waveform[:, i:i+frame_size]

			if frame.size(1) < frame_size:
				break

			yield self.extract_features(frame)

	def extract_features(self, frame: any):
		result = dict[str, any]

		for feature in self.features.items():
			result[feature[0]] = feature[1](frame)
		
		return result
