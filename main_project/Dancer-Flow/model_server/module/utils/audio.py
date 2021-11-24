import librosa

# Load Audio Recordings 
def Audio(wav):
    """ 
    Load an audio file as floating point time series.
    returns audio time series and its sampling rate
    : params pro_data amplitude 
    : params pro_sr sampling rate 
    """
    data, sr = librosa.load(wav)
    return data, sr

def Melody(data, sr):
    hop_size = 2205
    n_fft = 4410
    chroma = librosa.feature.chroma_stft(y=data, sr=sr, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft)
    return chroma