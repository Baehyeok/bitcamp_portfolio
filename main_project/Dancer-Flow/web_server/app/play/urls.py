from django.urls import path
from django.urls.resolvers import URLPattern
from .views import PlayView, OptionView, PreShareView
from django.conf.urls import url


urlpatterns = [
    path('', PlayView.as_view(), name='play'),
    path('option/', OptionView.as_view(), name='option'),
    path('api/options', OptionView.as_view(), name='options'),
    path('<str:play_id>/', PlayView.as_view(), name='playapi'),
    path('?pid=<str:play_id>', PlayView.as_view(), name='playapi'),
    path('preshare', PreShareView.as_view(), name='community_share'),

]  