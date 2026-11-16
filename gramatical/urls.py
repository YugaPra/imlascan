from django.conf import settings
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('home.urls')),
]
if settings.ENABLE_ADMIN:   # mati secara default di produksi (lihat settings.py)
    urlpatterns.insert(0, path('admin/', admin.site.urls))
