import librosa

#ffmpegのインストールが必要

a,sr = librosa.load("sample.mp3")

data = []
y = librosa.feature.mfcc(y=a, sr=sr)
data.append(y)
for i in range(len(data)):
    data[i] = sum(data[i])/len(data[i])

