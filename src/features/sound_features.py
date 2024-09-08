import torchaudio

def mfcc(frame):
    return torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={"n_mels": 64}
	)(frame)

def stft_spectrogram(frame):
    return torchaudio.transforms.Spectrogram()(frame)