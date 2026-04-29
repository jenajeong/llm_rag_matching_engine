from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/search/', include('search.urls')),
    path('api/v1/report/', include('report.urls')),
]
