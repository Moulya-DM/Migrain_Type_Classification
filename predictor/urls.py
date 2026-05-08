from django.urls import path
from . import views

urlpatterns = [

    path('', views.home),

    path('register/', views.register_user),

    path('login/', views.login_user),

    path('predict/', views.predict),
]