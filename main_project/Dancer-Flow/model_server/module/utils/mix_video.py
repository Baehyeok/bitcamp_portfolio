from moviepy.editor import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import  clips_array

def mix_video(user_video, target_video, time, output="result"):
    clip1 = VideoFileClip(user_video)
    clip2 = VideoFileClip(target_video)

    clip1 = clip1.resize(width=960, height=540)
    clip2 = clip2.resize(width=960, height=540)

    clip2 = clip2.subclip(time["start"], time["end"])

    final_clip = clips_array([[clip1,clip2]])

    final_clip.write_videofile("media/{}.mp4".format(output))