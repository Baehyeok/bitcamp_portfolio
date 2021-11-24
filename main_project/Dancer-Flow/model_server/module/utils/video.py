from moviepy.editor import *
from audio import *

def Video(start_point, end_point):
    clip = VideoFileClip("pro_clip")
    new_clip = clip.set_start(start_point)
    new_clip = clip.set_end(end_point)
    new_clip = new_clip.ipython_display(width=360, maxduration=300)
    return new_clip
