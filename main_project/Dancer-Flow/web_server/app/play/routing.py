from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/play/(?P<play_id>\w+)$', consumers.PlayConsumer.as_asgi()),
    # re_path(r'ws/play/\?pid=\w+', consumers.PlayConsumer.as_asgi()),
]