import moviepy.editor as mp

def Clip (video, output):
    clip = mp.VideoFileClip(video)
    clip.audio.write_audiofile(output)


