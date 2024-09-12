from torch.utils.data import DataLoader
from dataset.audio_dataset import AudioDataset
from dataset.features_extractor import FeaturesExtractor
from features.sound_features import *

# import test_tools.test_features

features = {
	'stft': stft_spectrogram,
	'stft9': stft_spectrogram_9,
	'fcwt': cwt_spectrogram,
	'slt': slt_spectrogram,
	#'mfcc': mfcc,
}

features_extractor = FeaturesExtractor(frame_size_ms=40, hop_length_ms=40, features=features, precomputed_dir='../data/precomputed')
audio_dataset = AudioDataset(dir='../data/output', features_extractor=features_extractor)

data_loader = DataLoader(audio_dataset, batch_size=1)

for batch in data_loader:
	labels_batch = batch['label']
	features = batch['features']
	pad = batch['pad']

	for feature_name, feature_data in features.items():
		pass