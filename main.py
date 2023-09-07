import pyaudio
import librosa
import IPython.display as ipd
import wave
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the audio parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 2900  # Sample rate (samples per second)
RECORD_SECONDS = 5  # Duration of the recording in seconds
OUTPUT_FILENAME = "output.wav"  # Name of the output WAV file

p = pyaudio.PyAudio()

# Create an audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=1024)

print("Recording...")

frames = []

# Record audio for the specified duration
for _ in range(0, int(RATE / 1024 * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)

print("Finished recording.")

# Convert the recorded audio frames to a NumPy array
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# Create a time array for the x-axis based on the sample rate
time = np.arange(0, len(audio_data)) / float(RATE)

# Plot the audio waveform
plt.figure(figsize=(12, 4))
plt.plot(time, audio_data, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Recorded Audio Waveform")
plt.grid()
plt.show()
# Stop and close the audio stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio instance
p.terminate()

# Save the recorded audio as a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
file = "output.wav"
ipd.Audio(file)
y, sr = librosa.load(file)
# Calculate mean frequency, standard deviation, and spectral centroid
mean_freq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))/10000
std_freq = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))/10000
centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))/1000

# Print the calculated features
print("Mean Frequency:", mean_freq)
print("Standard Deviation of Frequency:", std_freq)
print("Spectral Centroid:", centroid)

path="voice.csv"

csvfile=pd.read_csv(path)
subset = csvfile["meanfreq"][csvfile["meanfreq"] > mean_freq]
subset2 = csvfile["meanfreq"][csvfile["meanfreq"] < mean_freq]
print(subset,"\n suset2\n",subset2)