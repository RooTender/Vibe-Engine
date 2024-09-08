from torch.utils.data import DataLoader
from dataset.audio_dataset import AudioDataset
from dataset.features_extractor import FeaturesExtractor
from features.sound_features import *

features = {
	'mfcc': mfcc,
	'stft': stft_spectrogram
}

features_extractor = FeaturesExtractor(frame_size_ms=25, hop_length_ms=10, features=features)
audio_dataset = AudioDataset(dir='../data/output', features_extractor=features_extractor)

data_loader = DataLoader(audio_dataset, batch_size=1)

for batch in data_loader:
	labels_batch = batch['label']
	features = batch['features']

	for feature_name, feature_data in features.items():
		pass