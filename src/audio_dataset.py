import torchaudio
from torchvision.datasets import DatasetFolder
from features_extractor import FeaturesExtractor

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

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        waveform, sample_rate = self.loader(path)

        frames = list(self.feature_extractor.stream_audio_from_waveform(waveform, sample_rate))

        return {'features': frames, 'label': target}
