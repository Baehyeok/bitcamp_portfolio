
from .utils.audio import Audio, Melody
from .utils.clip import Clip
from .utils.dtw import Dtw
from .utils.mix_video import mix_video

def video__init__(user_video, target_video, output_name):
    user_audio_name = f"play_audio/{output_name}_wav1.wav" 
    play_audio_name = f"play_audio/{output_name}_wav2.wav"
    Clip(user_video, user_audio_name)
    Clip(target_video, play_audio_name)
    
    x1, sr = Audio(user_audio_name)
    x2, sr = Audio(play_audio_name)
    
    x1_chroma = Melody(x1, sr)
    x2_chroma = Melody(x2, sr)
    
    time = Dtw(x1_chroma, x2_chroma, sorted=sr)

    print("start time : ", time["start"])
    print("end time : ", time["end"])

    return time['start'], time['end'], user_audio_name, play_audio_name
    mix_video(user_video=user_video, target_video=target_video, time=time, output='sync_result')

if __name__ == '__main__':
    video__init__(
        user_video="sample_data/sample.mp4",
        target_video="sample_data/BTS-Dynamite1-3.mp4",
        output="sync_result"
    )