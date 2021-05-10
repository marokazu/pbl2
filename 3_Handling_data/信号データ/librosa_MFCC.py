import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
#ffmpegのインストールが必要

a,sr = librosa.load("sample.mp3")
data = []
y = librosa.feature.mfcc(y=a, sr=sr)
data.append(y)
for i in range(len(data)):
    data[i] = sum(data[i])/len(data[i])


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax1.plot(data[0])
ax2.plot(data[0])
ax3.plot(data[0])
ax1.set_xlim(150,770)
ax2.set_xlim(2900,3750)
ax3.set_xlim(3750,4700)

ax1.set_title('150-770(midium)')
ax2.set_title('2900-3750(low)')
ax3.set_title('3750-4700(high)')
fig.tight_layout()
plt.show()
