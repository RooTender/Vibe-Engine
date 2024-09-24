import os
import torch
from features.sound_features import fft_cache

class FeaturesExtractor():
    def __init__(self, frame_size_ms: int, hop_length_ms: int, features: dict[str, any] = None, precomputed_dir=None) -> None:
        self.frame_size_ms = frame_size_ms
        self.hop_length_ms = hop_length_ms
        self.features = features

        if precomputed_dir:
            os.makedirs(precomputed_dir, exist_ok=True)
            self.precomputed_dir = precomputed_dir

    def stream_audio(self, filepath: str, loader):
        # Extract the filename for use in caching
        filename = os.path.splitext(os.path.basename(filepath))[0]

        cached_features = {}

        # Load any precomputed features if they exist
        if self.precomputed_dir:
            for feature_name in self.features.keys():
                feature_filename = os.path.join(self.precomputed_dir, f"{feature_name}_{filename}.pt")

                if os.path.exists(feature_filename):
                    # Load both 'vector' and 'size' from the saved file
                    data = torch.load(feature_filename, weights_only=False)
                    cached_features[feature_name] = {
                        'vector': data['vector'],
                        'size': data['size']  # Load the 'size' key as well
                    }

        # If all features are cached and force_load is False, return them
        if len(cached_features) == len(self.features):
            return cached_features

        # Load audio
        waveform, sample_rate = loader(filepath)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        frame_size = int(self.frame_size_ms * sample_rate / 1000)
        hop_length = int(self.hop_length_ms * sample_rate / 1000)

        waveform = waveform / torch.max(torch.abs(waveform))

        features_info = cached_features.copy()  # Start with already cached features
        num_frames = (waveform.size(0) - frame_size) // hop_length + 1

        # Compute missing features for each frame
        for name, feature_fn in self.features.items():
            if name not in cached_features:
                first_frame = waveform[0:frame_size]
                feature_value = feature_fn(first_frame, sample_rate)
                feature_size = feature_value.reshape(-1).shape[0]

                # Initialize a 1D vector for the missing feature based on its individual size
                features_info[name] = {
                    'vector': torch.empty(num_frames * feature_size),  # Allocate 1D vector
                    'size': feature_size  # Store feature size
                }

                # Compute the feature for each frame
                for i in range(num_frames):
                    start_idx = i * hop_length
                    frame = waveform[start_idx:start_idx + frame_size]

                    if frame.size(0) < frame_size:
                        break

                    feature_value = feature_fn(frame, sample_rate).reshape(-1)

                    # Check if the feature size is consistent
                    if feature_value.shape[0] != feature_size:
                        raise ValueError(f"Size mismatch for feature '{name}': expected {feature_size}, got {feature_value.shape[0]}")

                    # Insert the computed features into the right section of the vector
                    features_info[name]['vector'][i * feature_size:(i + 1) * feature_size] = feature_value

        fft_cache.reset()

        # Save the newly computed features to disk if caching is enabled
        if self.precomputed_dir:
            for feature_name, feature_info in features_info.items():
                if feature_name not in cached_features:  # Save only new features
                    feature_filename = os.path.join(self.precomputed_dir, f"{feature_name}_{filename}.pt")
                    # Save both 'vector' and 'size' in the same file
                    torch.save({
                        'vector': feature_info['vector'],
                        'size': feature_info['size']
                    }, feature_filename)

        return features_info
