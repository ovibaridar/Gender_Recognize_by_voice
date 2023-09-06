import pyaudio
import librosa
import wave
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

path = "voice.csv"
voice_store = pd.read_csv(path)

# Set the audio parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 4100  # Sample rate (samples per second)
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

file = "C:/Users/Ovi/pythonProject/output.wav"
audio_data, sr = librosa.load(file)
stft = librosa.stft(audio_data)
magnitude = librosa.magphase(stft)[0]
mean_frequency = librosa.feature.spectral_centroid(S=magnitude)
mean_frequency_khz = mean_frequency / 1000
frequency_std = np.std(mean_frequency_khz)

print(frequency_std)

path = "voice.csv"
voice_store = pd.read_csv(path)
ste_male = voice_store["sd"].loc[voice_store["label"] == "male"]
ste_female = voice_store["sd"].loc[voice_store["label"] == "female"]

minsdmale = ste_male.min()
maxsdmale = ste_male.max()
maxsdfmale = ste_female.min()
maxsdfmale = ste_female.max()


print(minsdmale)
print(maxsdmale)




print(maxsdfmale)
print(maxsdfmale)

