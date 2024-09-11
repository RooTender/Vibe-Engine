import torch
import torchaudio
from features.sound_features import *
import matplotlib.pyplot as plt
import librosa

def extract_frame(audio_path, frame_size_ms: int, skip_frames: int):
	waveform, sample_rate = torchaudio.load(audio_path)
	
	frame_size = int(frame_size_ms * sample_rate / 1000)
	waveform = waveform / torch.max(torch.abs(waveform))

	frame = waveform[:, skip_frames:skip_frames + frame_size]
	time_extent = [0, frame.size(1) / sample_rate, 0, sample_rate / 2]

	return time_extent, frame, sample_rate

time_extent, frame, sample_rate = extract_frame(audio_path='../data/output/Anger/S001_001.wav', frame_size_ms=1000, skip_frames=10000)
frame = torch.mean(frame, dim=0)

# Plot the waveform of the extracted frame
time_axis = np.linspace(0, frame.size(0) / sample_rate, num=frame.size(0))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, frame.numpy())
plt.title("Waveform of the Extracted Frame")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.savefig('../artefacts/waveform.png')

# Now compute the STFT spectrogram using the provided function
stft_output = stft_spectrogram(frame, sample_rate)

plt.figure(figsize=(10, 4))
plt.imshow(torchaudio.transforms.AmplitudeToDB()(stft_output), aspect='auto', origin='lower', extent=time_extent)
plt.title("STFT Spectrogram")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.savefig('../artefacts/stft.png')

# Now compute the STFT with power 9 spectrogram using the provided function
stft_9_output = stft_spectrogram_9(frame, sample_rate)

plt.figure(figsize=(10, 4))
plt.imshow(torchaudio.transforms.AmplitudeToDB()(stft_9_output), aspect='auto', origin='lower', extent=time_extent)
plt.title("STFT Spectrogram")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.savefig('../artefacts/stft9.png')

# Now compute the fCWT spectrogram using the provided function
fcwt_output = cwt_spectrogram(frame, sample_rate)

plt.figure(figsize=(10, 4))
plt.imshow(np.flipud(np.abs(fcwt_output)), aspect='auto', origin='lower', extent=time_extent)
plt.title("fCWT Spectrogram")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.savefig('../artefacts/cwt.png')

# Now compute the Superlet spectrogram using the provided function
slt_output = slt_spectrogram(frame, sample_rate)

plt.figure(figsize=(10, 4))
plt.imshow(librosa.amplitude_to_db(np.abs(slt_output), ref=np.max), aspect='auto', origin='lower', extent=time_extent)
plt.title("SLT Spectrogram")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.savefig('../artefacts/slt.png')



# Create a figure with constrained layout and GridSpec for custom layout
fig, axes = plt.subplot_mosaic(
    [
        ["Waveform", "Waveform"],
        ["STFT", "STFT-9"],
        ["fCWT", "SLT"]
    ],
    figsize=(10, 15),
    constrained_layout=True
)

# Plot A: Waveform
axes["Waveform"].plot(time_axis, frame.numpy())
axes["Waveform"].set_title("Waveform of the Extracted Frame")
axes["Waveform"].set_xlabel("Time [sec]")
axes["Waveform"].set_ylabel("Amplitude")

# Plot B: STFT Spectrogram
axes["STFT"].imshow(torchaudio.transforms.AmplitudeToDB()(stft_output), aspect='auto', origin='lower', extent=time_extent)
axes["STFT"].set_title("STFT Spectrogram")
axes["STFT"].set_xlabel("Time [sec]")
axes["STFT"].set_ylabel("Frequency [Hz]")

# Plot C: STFT Spectrogram with Power 9
axes["STFT-9"].imshow(torchaudio.transforms.AmplitudeToDB()(stft_9_output), aspect='auto', origin='lower', extent=time_extent)
axes["STFT-9"].set_title("STFT with Power 9 Spectrogram")
axes["STFT-9"].set_xlabel("Time [sec]")
axes["STFT-9"].set_ylabel("Frequency [Hz]")

# Plot D: fCWT Spectrogram
axes["fCWT"].imshow(np.flipud(np.abs(fcwt_output)), aspect='auto', origin='lower', extent=time_extent)
axes["fCWT"].set_title("fCWT Spectrogram")
axes["fCWT"].set_xlabel("Time [sec]")
axes["fCWT"].set_ylabel("Frequency [Hz]")

# Plot E: SLT Spectrogram
axes["SLT"].imshow(librosa.amplitude_to_db(np.abs(slt_output), ref=np.max), aspect='auto', origin='lower', extent=time_extent)
axes["SLT"].set_title("SLT Spectrogram")
axes["SLT"].set_xlabel("Time [sec]")
axes["SLT"].set_ylabel("Frequency [Hz]")

# Save the figure with all plots
plt.savefig('../artefacts/all_plots.png', dpi=300)
exit()