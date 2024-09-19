# image_processing/urls.py

from django.urls import path
from .views import analysis_results_view, process_image

urlpatterns = [

    
   
    path('process-image/', process_image, name='process_image'),
]
