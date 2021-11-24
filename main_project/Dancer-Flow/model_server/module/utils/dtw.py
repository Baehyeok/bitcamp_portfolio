import numpy as np
import librosa
import librosa.core
import librosa.display

# Load Audio Recordings 
# First, we loaded audio from the professional dancer which is target video 

def Dtw(x1_chroma,x2_chroma, sorted):
    """
    Computes Dynamic Time Wariping DTW) of two sequences.

    """
    hop_size = 2205

    D, wp = librosa.sequence.dtw(x1_chroma, x2_chroma, subseq=True)
    wp_s = np.asarray(wp) * int(hop_size) / sorted
    start_point = wp_s[:, 1][-1]
    # end_point = wp_s[:, 1][0] - start_point
    end_point = wp_s[:, 1][0]
    result = {
        "start" : start_point,
        "end" : end_point
    }
    return result
