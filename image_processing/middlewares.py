# middlewares.py
from django.utils.deprecation import MiddlewareMixin

class XFrameOptionsMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        if request.path.startswith('/report/'):
            response.headers.pop('X-Frame-Options', None)
        return response
