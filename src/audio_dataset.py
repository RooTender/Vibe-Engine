import os
from torch.utils.data import Dataset
from typing import Dict, Any

from features_extractor import FeaturesExtractor


class AudioDataset(Dataset[Any]):
    def __init__(self, dir: str, features_extractor: FeaturesExtractor) -> None:
        self.samples = []
        self.feature_extractor = features_extractor

        for root_path, _, file_names in os.walk(dir):
            for file_name in file_names:
                full_path = os.path.join(root_path, file_name)

                if self.is_audio_file(full_path):
                    self.samples.append(full_path)
    
    def is_audio_file(self, file_name: str) -> bool:
        """Check if the file is an audio file based on its extension."""

        _, ext = os.path.splitext(file_name)
        return ext.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frames = list(self.feature_extractor.stream_audio(self.samples[index]))
        return { 'features': frames }
