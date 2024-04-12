import os
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, Any


class AudioDataset(Dataset[Any]):
    def __init__(self, dir: str) -> None:
        self.samples = []

        for root_path, _, file_names in os.walk(dir):
            for file_name in file_names:
                full_path = os.path.join(root_path, file_name)
                dir_name = os.path.basename(root_path)
                self.samples.append((full_path, dir_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        audio_file, label = self.samples[index]
        waveform, _ = torchaudio.load(audio_file)

        return {"waveform": waveform, "label": label}
