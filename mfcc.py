import numpy as np
import librosa
import matplotlib.pyplot as plt

#Loading a trumpet audio file
audio_path = librosa.example('trumpet')
y, sr = librosa.load(audio_path) #y = audio time series, sr = sampling rate of y

plt.figure(figsize = (14, 5))
plt.plot(y)
plt.title('Waveform of the Audio Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Computing the full MFCC of an audio file:
# 1. Pre-emphasize the signal
# 2. Framing
# 3. Windowing
# 4. FFT (Fasr Fourier Transform)
# 5. Mel-filterbank
# 6. Logarithm - take log of output from Mel-filterbank
# 7. DCT (Discrete Cosine Transform)

#Pre-emphasis -> amplifies higher frequencies to balance the spectrum



