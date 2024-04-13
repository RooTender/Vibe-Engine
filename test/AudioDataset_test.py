import pytest
from unittest.mock import patch, MagicMock
from src.AudioDataset import AudioDataset

FAKE_ROOT_DIR = "/fake/dir"

@pytest.fixture
def mock_os_walk():
    with patch("os.walk") as mockwalk:
        mockwalk.return_value = [
            (f"{FAKE_ROOT_DIR}/emotion1", ("subdirs",), ("audio1.wav", "audio2.wav")),
            (f"{FAKE_ROOT_DIR}/emotion2", ("subdirs",), ("audio3.wav",)),
        ]
        yield mockwalk


@pytest.fixture
def mock_torchaudio_load():
    with patch("torchaudio.load") as mockload:
        mock_waveform = MagicMock()
        mock_sample_rate = 16000
        mockload.return_value = (mock_waveform, mock_sample_rate)
        yield mockload


def test_dataset_initialization(mock_os_walk, mock_torchaudio_load):
    dataset = AudioDataset(FAKE_ROOT_DIR)

    assert len(dataset.samples) == 3


def test_getitem_returns_correct_structure(mock_os_walk, mock_torchaudio_load):
    dataset = AudioDataset(FAKE_ROOT_DIR)
    sample = dataset[0]

    assert "waveform" in sample and "label" in sample
    assert sample["label"] == "emotion1"


def test_getitem_applies_features(mock_os_walk, mock_torchaudio_load):
    feature = MagicMock(return_value="feature_result")
    features = [("feature1", feature)]
    dataset = AudioDataset(FAKE_ROOT_DIR, features=features)
    sample = dataset[0]

    feature.assert_called_once_with(sample["waveform"])

    assert "feature1" in sample
    assert sample["feature1"] == "feature_result"


def test_getitem_with_multiple_features(mock_os_walk, mock_torchaudio_load):
    feature = MagicMock(return_value="feature_result")
    feature2 = MagicMock(return_value="feature2_result")
    features = [("feature1", feature), ("feature2", feature2)]
    dataset = AudioDataset(FAKE_ROOT_DIR, features=features)
    sample = dataset[0]

    feature.assert_called_once_with(sample["waveform"])
    feature2.assert_called_once_with(sample["waveform"])

    assert "feature1" in sample and "feature2" in sample
    assert sample["feature1"] == "feature_result"
    assert sample["feature2"] == "feature2_result"


def test_getitem_handles_file_load_failure(mock_os_walk, mock_torchaudio_load):
    mock_torchaudio_load.side_effect = Exception("Failed to load")
    dataset = AudioDataset(FAKE_ROOT_DIR)

    with pytest.raises(Exception):
        dataset[0]
