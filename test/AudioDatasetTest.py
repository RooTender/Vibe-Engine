import pytest
from unittest.mock import patch, MagicMock
from src.AudioDataset import AudioDataset


@pytest.fixture
def mock_os_walk():
    with patch("os.walk") as mockwalk:
        mockwalk.return_value = [
            ('/fake/dir/emotion1', ('subdirs',), ('audio1.wav', 'audio2.wav')),
            ('/fake/dir/emotion2', ('subdirs',), ('audio3.wav',))
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
    dataset = AudioDataset("/fake/dir")
    assert len(dataset.samples) == 3


def test_getitem_returns_correct_structure(mock_os_walk, mock_torchaudio_load):
    dataset = AudioDataset("/fake/dir")
    sample = dataset[0]
    assert "waveform" in sample and "label" in sample
    assert sample["label"] == "emotion1"


def test_getitem_handles_file_load_failure(mock_os_walk, mock_torchaudio_load):
    mock_torchaudio_load.side_effect = Exception("Failed to load")
    dataset = AudioDataset("/fake/dir")
    with pytest.raises(Exception):
        dataset[0]
