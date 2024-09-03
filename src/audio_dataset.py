import os
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, Any, List, Callable, Tuple


def is_audio_file(file_name: str) -> bool:
    """Check if the file is an audio file based on its extension."""

    _, ext = os.path.splitext(file_name)
    return ext.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

def preprocess_and_save(input_dir: str, output_dir: str, features: List[Tuple[str, Callable[[Any], Any]]] | None = None):
    input_dir = f'../data/{input_dir}'
    output_dir = f'../data/{output_dir}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        return

    for root_path, _, file_names in os.walk(input_dir):
        for file_name in file_names:
            if is_audio_file(file_name):
                full_path = os.path.join(root_path, file_name)
                dir_name = os.path.basename(root_path)
                
                # Load the audio file
                waveform, _ = torchaudio.load(full_path)
                
                # Compute features
                feature_dict = {"waveform": waveform, "label": dir_name}
                if features:
                    for name, feature in features:
                        feature_dict[name] = feature(waveform)
                
                # Save the feature_dict as a .pt file (you can also use .npy or other formats)
                save_path = os.path.join(output_dir, f"{file_name}.pt")
                torch.save(feature_dict, save_path)

class AudioDataset(Dataset[Any]):
    def __init__(self, dir: str) -> None:
        self.samples = []
        for root_path, _, file_names in os.walk(dir):
            for file_name in file_names:
                if file_name.endswith('.pt'):  # Assuming .pt is the format used for saving
                    full_path = os.path.join(root_path, file_name)
                    self.samples.append(full_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Load the precomputed features from the .pt file
        data = torch.load(self.samples[index])
        return data