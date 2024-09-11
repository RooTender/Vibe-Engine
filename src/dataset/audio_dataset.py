import torchaudio
from torchvision.datasets import DatasetFolder
from .features_extractor import FeaturesExtractor

class AudioDataset(DatasetFolder):
	def __init__(self, dir: str, features_extractor: FeaturesExtractor):
		super().__init__(
			root=dir, 
			loader=self.audio_loader, 
			extensions=('.wav', '.mp3', '.flac', '.ogg', '.m4a')
		)
		self.feature_extractor = features_extractor

	def audio_loader(self, path: str):
		waveform, sample_rate = torchaudio.load(path)
		return waveform, sample_rate
	
	def map_emotion_to_pad(self, emotion: str):
		emotion_to_pad = {
			"anger": [0.2, 0.8, 0.6],
			"boredom": [0.1, 0.3, 0.3],
			"disgust": [0.3, 0.7, 0.5],
			"fear": [0.1, 0.9, 0.3],
			"happiness": [0.9, 0.7, 0.8],
			"interest": [0.7, 0.6, 0.7],
			"sadness": [0.1, 0.3, 0.2],
			"unsure": [0.5, 0.5, 0.5] # Unsure means probably undefined
		}
		
		return emotion_to_pad.get(emotion, [0.5, 0.5, 0.5])

	def __getitem__(self, index: int):
		path, _ = self.samples[index]
		waveform, sample_rate = self.loader(path)

		label = str.lower(path.split('/')[-2])
		features_info = self.feature_extractor.stream_audio(waveform, sample_rate)
		pad_values = self.map_emotion_to_pad(label)

		return {
			'label': label, 
			'features': features_info,
			'pad': pad_values
		}
