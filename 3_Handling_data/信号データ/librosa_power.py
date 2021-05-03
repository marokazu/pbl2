import librosa
import numpy as np

#ffmpegのインストールが必要

a, sr = librosa.load("sample.mp3")

y = np.fft.fft(a)
amp = np.abs(y)
power = amp ** 2
x = np.linspace(0, sr, len(power))

