from django.urls import path
from .views import home
from .views import chatbot_response

urlpatterns = [
    path('', home, name="home"),
    path("chatbot/", chatbot_response, name="chatbot_response"),
]
