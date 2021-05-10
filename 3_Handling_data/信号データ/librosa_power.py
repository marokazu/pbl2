import librosa
import numpy as np
import matplotlib.pyplot as plt

#ffmpegのインストールが必要

def power_(n, sr, smp):
    y = np.fft.fft(n)
    amp = np.abs(y)
    power = amp ** 2
    x = np.linspace(0, sr, len(power))
    freqs = np.fft.fftfreq(smp, 1/sr)
    return power, freqs

a, sr_a = librosa.load("sample_1.mp3")
b, sr_b = librosa.load("sample_2.mp3")
c, sr_c = librosa.load("sample_3.mp3")
p_a, f_a =power_(a, sr_a, 336960)
p_b, f_b =power_(b, sr_b, 405504)
p_c, f_c =power_(c, sr_c, 419904)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

ax1.plot(f_a, p_a, color="blue", label="sample1")
ax2.plot(f_b, p_b, color="green", label="sample2")
ax3.plot(f_c, p_c, color="red", label="sample3")
ax1.set_xlim(0,1000)
ax2.set_xlim(0,1000)
ax3.set_xlim(0,1000)
ax1.set_title('sample_1(midium)')
ax2.set_title('sample_2(low)')
ax3.set_title('sample_3(high)')

fig.tight_layout()
plt.show()
