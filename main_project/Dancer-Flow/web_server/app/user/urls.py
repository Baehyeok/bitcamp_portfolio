from django.urls import path
from . import views
from .views import UserView, UsernameValidationView, EmailValidationView
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    path('', UserView.as_view(), name='user'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('validate-username', csrf_exempt(UsernameValidationView.as_view()), name = "validate-username"),
    path('validate-email', csrf_exempt(EmailValidationView.as_view()), name = "validate_email"),
    path('<str:user_id>/', UserView.as_view(), name='user'),
    
]