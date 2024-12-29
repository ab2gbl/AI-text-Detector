
from django.urls import path,include
from . import views

urlpatterns = [
    path('detect/', views.Detect.as_view()),
    path('', views.Page),
]
