from django.contrib import admin
from django.urls import path
from image_processing import views  # Import your views from the image_processing app

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),  # Homepage URL pattern
    path('process-image/', views.process_image, name='process_image'),  # Process image URL pattern
    path('report/', views.report, name='report'),
]

# Serve media files during development
from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
