from django.db import models
from django.db.models import fields
from django.db.models.base import Model
# from django.db import models
# from djongo import models
from django import forms
# Create your models here.
from server.settings import MEDIA_DIR
import os
from pathlib import Path

# class Song(models.Model):
#     title = models.TextField()
#     artist = models.TextField()
#     release_date = models.TextField()
#     file_path = models.TextField()
#     sound_length = models.TextField()


# class Option(models.Model):
#     MODE = (
#         ('PT', 'Practice'),
#         ('RD', 'Random'),
#     )
#     UPLOAD = (
#         ('UP', 'Upload'),
#         ('RT', 'RealTime'),
#     )
#     mode = models.TextField(max_length=4, blank=False, choices=MODE)
#     upload = models.TextField(max_length=4, blank=False, choices=UPLOAD)
#     # upload_file_path = models.TextField()
#     songs = models.TextField(blank=False)
#     video = models.FileField(upload_to='user_videos', blank=True, null=True)

# class OptionForm(forms.ModelForm):
#     class Meta:
#         model = Option
#         fields = ('mode', 'upload', 'video')       
#         def __init__(self, *args, **kwargs):
#             super(Option, self).__init__(*args, **kwargs)
#             self.fields['video'].reqired = False

# class OptionSong(models.Model):
#     option_id = models.ForeignKey(Option, on_delete=models.CASCADE)
#     songs = models.ForeignKey(Song, on_delete=models.CASCADE)

# class Play(models.Model):
#     # option = models.ForeignKey(Option, on_delete=models.CASCADE)

#     datetime = models.TextField()
#     score = models.IntegerField(default=0)
#     video = models.FileField(upload_to='play_videos', blank=True)