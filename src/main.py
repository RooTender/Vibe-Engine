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
    features_batch = batch['features']
    labels_batch = batch['label']

    print(f"Batch size: {len(features_batch)}, Labels: {labels_batch}")

    for features in features_batch:
        for feature_name, feature_value in features.items():
            pass
