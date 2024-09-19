import os
import shutil
from django.conf import settings
import numpy as np
import cv2
import tifffile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64


# # Define paths for uploaded and processed images
# UPLOAD_DIR = 'uploaded_images/'
# PROCESSED_DIR = 'processed_images/'

# # Ensure the directories exist
# os.makedirs(os.path.join(settings.MEDIA_ROOT, UPLOAD_DIR), exist_ok=True)
# os.makedirs(os.path.join(settings.MEDIA_ROOT, PROCESSED_DIR), exist_ok=True)



# def norm_gray(grayscale, imax):
#     """Normalizes uint8 image to value."""
#     assert grayscale.dtype == np.uint8, "Your Image is not of type np.uint8"
#     grayscale = (grayscale.astype(np.float64) / float(imax)) * 255.
#     num_of_sat = np.sum(grayscale > 255)
#     print(num_of_sat, 'number of saturated pixels.')
#     grayscale[grayscale > 255] = 255
#     return grayscale.astype(np.uint8)

# def create_LUT(rgb):
#     """Create a lookup table for coloring."""
#     r = rgb[0] / 256.0
#     g = rgb[1] / 256.0
#     b = rgb[2] / 256.0
#     return np.array([np.linspace(0, r, 256), np.linspace(0, g, 256), np.linspace(0, b, 256)]).T

# def add_scale_bar(img, origin, swidth, sheight):
#     orix = int(origin[0])
#     oriy = int(origin[1])
#     swidth = int(np.round(swidth, 0))
#     sheight = int(np.round(sheight, 0))
    
#     # Check if the coordinates are within the image dimensions
#     if oriy - sheight >= 0 and orix + swidth <= img.shape[1]:
#         img[oriy-sheight:oriy, orix:orix+swidth, :] = 255
#     else:
#         print("Scale bar coordinates are out of bounds.")
#     return img

# def merge_imgs(imgs):
#     assert len(imgs) > 1, "imgs should be an array with at least two images"
#     merged = np.zeros_like(imgs[0], dtype=np.float64)
#     for img in imgs:
#         merged += img.astype(np.float64)
#     merged[merged > 255] = 255
#     return merged.astype(np.uint8)


# def detect_edges(image, algorithm):
#     """Apply edge detection to the image based on the selected algorithm."""
#     # Ensure the image is in grayscale format
#     if len(image.shape) == 3:  # If the image has 3 channels, convert to grayscale
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     if algorithm == 'canny':
#         return cv2.Canny(image, 100, 200)
#     elif algorithm == 'sobel':
#         sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
#         sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
#         return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
#     elif algorithm == 'laplacian':
#         return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
#     else:
#         raise ValueError("Unknown edge detection algorithm")

# def overlay_edges_on_image(color_img, edges_img):
#     """Overlay edge detection results on color image."""
#     # Convert color image to grayscale
#     color_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
#     # Create a colored version of the edges for better visibility
#     edges_colored = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR)
    
#     # Overlay edges on the color image
#     overlay = cv2.addWeighted(color_img, 0.7, edges_colored, 0.3, 0)
#     return overlay
# def apply_LUT(grayscale, lut):
#     """Apply LUT to 2-D numpy array. Returns RGB 3-D numpy array."""
#     assert grayscale.dtype == np.uint8, "Your Image is not of type np.uint8"
#     height, width = grayscale.shape
#     rgb = lut[grayscale.flatten()].reshape(height, width, 3)
#     return (rgb * 255).astype(np.uint8)


# def delete_old_images(upload_dir, processed_dir):
#     """Delete old images from the upload and processed directories."""
#     if os.path.exists(upload_dir):
#         shutil.rmtree(upload_dir)
#         print("successfull deleted!")
#     if os.path.exists(processed_dir):
#         shutil.rmtree(processed_dir)
#         print("successfull deleted the processed_dir")
#     os.makedirs(upload_dir)
#     os.makedirs(processed_dir)


# def convert_to_tiff(image_path):
#     # Open the image using Pillow
#     img = Image.open(image_path)
    
#     # Convert the image to TIFF format
#     tiff_image_path = os.path.splitext(image_path)[0] + ".tiff"
#     img.save(tiff_image_path, format='TIFF')
    
#     return tiff_image_path

# @csrf_exempt
# def process_image(request):
#     if request.method == 'POST':
#         image_file = request.FILES['image']
#         algorithm = request.POST.get('algorithm', 'canny')

#          # Define paths for uploaded and processed images
#         upload_dir = default_storage.path('uploaded_images')
#         processed_dir = default_storage.path('processed_images')

#         # Delete old images before processing new ones
#         delete_old_images(upload_dir, processed_dir)

#         # Save the uploaded file in the "uploaded_images" directory
#         upload_path = os.path.join(UPLOAD_DIR, image_file.name)
#         file_name = default_storage.save(upload_path, ContentFile(image_file.read()))
#         image_path = default_storage.path(file_name)

#        # Load the image
#         img = tifffile.imread(image_path)
#         scale_of_pixel = 9.5974  # Pixels per um
#         scale = 10  # um, how big I want my scale to be.
#         swidth = scale_of_pixel * scale  # Final scale in pixels
#         orix = img.shape[2] - 20 - swidth  # X position for the scale bar
#         oriy = img.shape[1] - 20  # Y position for the scale bar

#         # Create LUTs with alternative colors
#         cmap_c_dat = create_LUT((0, 255, 255))  # Cyan
#         cmap_r_dat = create_LUT((255, 0, 0))    # Red
#         cmap_y_dat = create_LUT((255, 222, 75)) # Yellow
#         cmap_g_dat = create_LUT((0, 255, 0))    # Green
#         # cmap_p_dat = create_LUT(128, 0, 128)  # RGB for Purple

#         # Normalize images
#         gray_c = norm_gray(img[0, :, :], 158)
#         gray_y = norm_gray(img[1, :, :], 50)
#         gray_g = norm_gray(img[2, :, :], 158)
#         gray_r = norm_gray(img[3, :, :], 158)
#         # gray_p = norm_gray(img[3, :, :], 158)

#         # Apply LUTs
#         rgb_c = apply_LUT(gray_c, cmap_c_dat)
#         rgb_y = apply_LUT(gray_y, cmap_y_dat)
#         rgb_g = apply_LUT(gray_g, cmap_g_dat)
#         rgb_r = apply_LUT(gray_r, cmap_r_dat)
#         # rgb_p = apply_LUT(gray_p, cmap_p_dat)

#         # Apply edge detection based on selected algorithm
#         edges_c = detect_edges(gray_c, algorithm)
#         edges_y = detect_edges(gray_y, algorithm)
#         edges_g = detect_edges(gray_g, algorithm)
#         edges_r = detect_edges(gray_r, algorithm)
#         # edges_p = detect_edges(gray_p, algorithm)

#         # Overlay edges on original color images
#         overlay_c = overlay_edges_on_image(rgb_c, edges_c)
#         overlay_y = overlay_edges_on_image(rgb_y, edges_y)
#         overlay_g = overlay_edges_on_image(rgb_g, edges_g)
#         overlay_r = overlay_edges_on_image(rgb_r, edges_r)
#         # overlay_p = overlay_edges_on_image(rgb_p, edges_p)

#         # Merge images
#         merged = merge_imgs([overlay_c, overlay_r])

#         # Add scale bars
#         overlay_c = add_scale_bar(overlay_c, (orix, oriy), swidth, 10)
#         overlay_y = add_scale_bar(overlay_y, (orix, oriy), swidth, 10)
#         overlay_g = add_scale_bar(overlay_g, (orix, oriy), swidth, 10)
#         overlay_r = add_scale_bar(overlay_r, (orix, oriy), swidth, 10)
#         # overlay_p = add_scale_bar(overlay_p, (orix, oriy), swidth, 10)
#         merged = add_scale_bar(merged, (orix, oriy), swidth, 10)

#          # Convert images to a format suitable for rendering in HTML
#         def save_img_to_storage(img, name):
#             processed_file_name = os.path.join(PROCESSED_DIR, algorithm +'_'+'processed_' + name )
#             processed_image_path = default_storage.save(processed_file_name, ContentFile(cv2.imencode('.png', img)[1].tobytes()))
#             return default_storage.url(processed_image_path)
        
        

#         results = [
#             {'image_url': save_img_to_storage(overlay_c, 'cyan_overlay.png'), 'algorithm': 'Cyan Channel with Edges'},
#             {'image_url': save_img_to_storage(overlay_y, 'yellow_overlay.png'), 'algorithm': 'Yellow Channel with Edges'},
#             {'image_url': save_img_to_storage(overlay_g, 'green_overlay.png'), 'algorithm': 'Green Channel with Edges'},
#             {'image_url': save_img_to_storage(overlay_r, 'red_overlay.png'), 'algorithm': 'Red Channel with Edges'},
#             {'image_url': save_img_to_storage(merged, 'merged.png'), 'algorithm': 'Merged Image with Edges'}
#         ]

#         # Render the results in the result.html template
#         return render(request, 'image_processing/result.html', {'results': results, 'selected_algorithm': algorithm})

#     return JsonResponse({'error': 'Invalid request method!'}, status=405)

# import os
# import cv2
# import numpy as np
# from PIL import Image
# from django.conf import settings
# from django.core.files.base import ContentFile
# from django.core.files.storage import default_storage
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt
# from django.http import JsonResponse
# import shutil

# # Define paths for uploaded and processed images
# UPLOAD_DIR = 'uploaded_images/'
# PROCESSED_DIR = 'processed_images/'

# # Ensure the directories exist
# os.makedirs(os.path.join(settings.MEDIA_ROOT, UPLOAD_DIR), exist_ok=True)
# os.makedirs(os.path.join(settings.MEDIA_ROOT, PROCESSED_DIR), exist_ok=True)

# def norm_gray(grayscale, imax):
#     grayscale = (grayscale.astype(np.float64) / float(imax)) * 255.
#     grayscale[grayscale > 255] = 255
#     return grayscale.astype(np.uint8)

# def create_LUT(rgb):
#     return np.array([np.linspace(0, c / 256.0, 256) for c in rgb]).T

# def add_scale_bar(img, origin, swidth, sheight):
#     orix, oriy = int(origin[0]), int(origin[1])
#     swidth, sheight = int(np.round(swidth, 0)), int(np.round(sheight, 0))
#     if 0 <= oriy - sheight < img.shape[0] and 0 <= orix + swidth <= img.shape[1]:
#         img[oriy-sheight:oriy, orix:orix+swidth, :] = 255
#     return img

# def merge_imgs(imgs):
#     merged = np.sum(imgs, axis=0, dtype=np.float64)
#     return np.clip(merged, 0, 255).astype(np.uint8)

# def detect_edges(image, algorithm):
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if algorithm == 'canny':
#         return cv2.Canny(image, 100, 200)
#     elif algorithm == 'sobel':
#         sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
#         sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
#         return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
#     elif algorithm == 'laplacian':
#         return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
#     else:
#         raise ValueError("Unknown edge detection algorithm")

# def overlay_edges_on_image(color_img, edges_img):
#     edges_colored = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR)
#     return cv2.addWeighted(color_img, 0.7, edges_colored, 0.3, 0)

# def apply_LUT(grayscale, lut):
#     return (lut[grayscale] * 255).astype(np.uint8)

# def delete_old_images(upload_dir, processed_dir):
#     for dir_path in [upload_dir, processed_dir]:
#         if os.path.exists(dir_path):
#             shutil.rmtree(dir_path)
#         os.makedirs(dir_path)

# def safe_delete_directory(directory):
#     """Safely delete a directory and all its contents."""
#     full_path = os.path.join(settings.MEDIA_ROOT, directory)
#     if os.path.exists(full_path):
#         shutil.rmtree(full_path, ignore_errors=True)
#     os.makedirs(full_path, exist_ok=True)

# @csrf_exempt
# def process_image(request):
#     if request.method == 'POST':
#         image_file = request.FILES['image']
#         algorithm = request.POST.get('algorithm', 'canny')

#         # Clean up old directories
#         safe_delete_directory(UPLOAD_DIR)
#         safe_delete_directory(PROCESSED_DIR)

#         # Save uploaded file
#         upload_path = os.path.join(UPLOAD_DIR, image_file.name)
#         file_name = default_storage.save(upload_path, image_file)
        
#         # Open image using PIL
#         with default_storage.open(file_name) as f:
#             img = Image.open(f)
#             if img.mode != 'RGB':
#                 img = img.convert('RGB')
#             img_np = np.array(img)

#         height, width, channels = img_np.shape

#         scale_of_pixel = 9.5974
#         scale = 10
#         swidth = scale_of_pixel * scale
#         orix = width - 20 - swidth
#         oriy = height - 20

#         cmap_colors = [
#             ((0, 255, 255), 'cyan'),
#             ((255, 222, 75), 'yellow'),
#             ((0, 255, 0), 'green'),
#             ((255, 0, 0), 'red'),
#         ]

#         results = []

#         for i in range(min(channels, len(cmap_colors))):
#             rgb, color_name = cmap_colors[i]
#             cmap = create_LUT(rgb)
            
#             gray = norm_gray(img_np[:,:,i], np.max(img_np[:,:,i]))
#             rgb_img = apply_LUT(gray, cmap)
            
#             edges = detect_edges(gray, algorithm)
#             overlay = overlay_edges_on_image(rgb_img, edges)
#             overlay = add_scale_bar(overlay, (orix, oriy), swidth, 10)

#             processed_file_name = f'{algorithm}_{color_name}_overlay.png'
#             processed_path = os.path.join(PROCESSED_DIR, processed_file_name)
            
#             # Save processed image using default_storage
#             buffer = cv2.imencode('.png', overlay)[1].tobytes()
#             processed_image_path = default_storage.save(processed_path, ContentFile(buffer))
            
#             results.append({
#                 'image_url': default_storage.url(processed_image_path),
#                 'algorithm': f'{color_name.capitalize()} Channel with Edges'
#             })

#         # Create and save merged image
#         if channels > 1:
#             merged_images = []
#             for result in results:
#                 with default_storage.open(result['image_url'].replace(settings.MEDIA_URL, '')) as f:
#                     merged_images.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1))
            
#             merged = merge_imgs(merged_images)
#             merged = add_scale_bar(merged, (orix, oriy), swidth, 10)
            
#             merged_file_name = f'{algorithm}_merged.png'
#             merged_path = os.path.join(PROCESSED_DIR, merged_file_name)
            
#             buffer = cv2.imencode('.png', merged)[1].tobytes()
#             merged_image_path = default_storage.save(merged_path, ContentFile(buffer))
            
#             results.append({
#                 'image_url': default_storage.url(merged_image_path),
#                 'algorithm': 'Merged Image with Edges'
#             })

#         return render(request, 'image_processing/result.html', {'results': results, 'selected_algorithm': algorithm})

#     return JsonResponse({'error': 'Invalid request method!'}, status=405)



import os
import cv2
import numpy as np
from PIL import Image
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import shutil

# Define paths for uploaded and processed images
UPLOAD_DIR = 'uploaded_images/'
PROCESSED_DIR = 'processed_images/'

# Ensure the directories exist
os.makedirs(os.path.join(settings.MEDIA_ROOT, UPLOAD_DIR), exist_ok=True)
os.makedirs(os.path.join(settings.MEDIA_ROOT, PROCESSED_DIR), exist_ok=True)

def norm_gray(grayscale, imax):
    grayscale = (grayscale.astype(np.float64) / float(imax)) * 255.
    grayscale[grayscale > 255] = 255
    return grayscale.astype(np.uint8)

def create_LUT(rgb):
    return np.array([np.linspace(0, c / 256.0, 256) for c in rgb]).T

def add_scale_bar(img, origin, swidth, sheight):
    orix, oriy = int(origin[0]), int(origin[1])
    swidth, sheight = int(np.round(swidth, 0)), int(np.round(sheight, 0))
    if 0 <= oriy - sheight < img.shape[0] and 0 <= orix + swidth <= img.shape[1]:
        img[oriy-sheight:oriy, orix:orix+swidth, :] = 255
    return img

def merge_imgs(imgs):
    merged = np.sum(imgs, axis=0, dtype=np.float64)
    return np.clip(merged, 0, 255).astype(np.uint8)

def detect_edges(image, algorithm):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if algorithm == 'canny':
        return cv2.Canny(image, 100, 200)
    elif algorithm == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    elif algorithm == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
    else:
        raise ValueError("Unknown edge detection algorithm")

def overlay_edges_on_image(color_img, edges_img):
    edges_colored = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(color_img, 0.7, edges_colored, 0.3, 0)

def apply_LUT(grayscale, lut):
    return (lut[grayscale] * 255).astype(np.uint8)

def add_border(img, border_size, border_color):
    img_with_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                          cv2.BORDER_CONSTANT, value=border_color)
    return img_with_border

def sharpen_edges(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def calculate_accuracy(original_image, processed_image):
    # Convert images to grayscale if they are not already
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Resize images to the same dimensions
    original_gray = cv2.resize(original_gray, (processed_gray.shape[1], processed_gray.shape[0]))
    
    # Calculate accuracy
    accuracy = np.sum(original_gray == processed_gray) / original_gray.size * 10  
    
    return accuracy

def safe_delete_directory(directory):
    """Safely delete a directory and all its contents."""
    full_path = os.path.join(settings.MEDIA_ROOT, directory)
    if os.path.exists(full_path):
        shutil.rmtree(full_path, ignore_errors=True)
    os.makedirs(full_path, exist_ok=True)

@csrf_exempt
def process_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        algorithm = request.POST.get('algorithm', 'canny')

        if not image_file:
            return JsonResponse({'error': 'No image file provided!'}, status=400)

        # Clean up old directories
        safe_delete_directory(UPLOAD_DIR)
        safe_delete_directory(PROCESSED_DIR)

        # Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, image_file.name)
        file_name = default_storage.save(upload_path, image_file)
        
        # Open image using PIL
        with default_storage.open(file_name) as f:
            img = Image.open(f)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_np = np.array(img)

        height, width, channels = img_np.shape

        scale_of_pixel = 9.5974
        scale = 10
        swidth = scale_of_pixel * scale
        orix = width - 20 - swidth
        oriy = height - 20

        cmap_colors = [
            ((0, 255, 255), 'cyan'),
            ((255, 222, 75), 'yellow'),
            ((0, 255, 0), 'green'),
            ((255, 0, 0), 'red'),
        ]

        results = []

        for i in range(min(channels, len(cmap_colors))):
            rgb, color_name = cmap_colors[i]
            cmap = create_LUT(rgb)
            
            gray = norm_gray(img_np[:,:,i], np.max(img_np[:,:,i]))
            rgb_img = apply_LUT(gray, cmap)
            
            edges = detect_edges(gray, algorithm)
            overlay = overlay_edges_on_image(rgb_img, edges)
            overlay = add_scale_bar(overlay, (orix, oriy), swidth, 10)

            # Add border and sharpen edges
            overlay_with_border = add_border(overlay, border_size=10, border_color=(255, 255, 255))
            sharpened_overlay = sharpen_edges(overlay_with_border)
            
            processed_file_name = f'{algorithm}_{color_name}_overlay.png'
            processed_path = os.path.join(PROCESSED_DIR, processed_file_name)
            
            # Save processed image using default_storage
            buffer = cv2.imencode('.png', sharpened_overlay)[1].tobytes()
            processed_image_path = default_storage.save(processed_path, ContentFile(buffer))
            
            # Calculate accuracy
            original_image_path = os.path.join(settings.MEDIA_ROOT, upload_path)
            original_img = cv2.imread(original_image_path)
            processed_img = cv2.imread(os.path.join(settings.MEDIA_ROOT, processed_path))
            accuracy = calculate_accuracy(original_img, processed_img)

            results.append({
                'image_url': default_storage.url(processed_image_path),
                'algorithm': f'{color_name.capitalize()} Channel with Edges',
                'accuracy': accuracy
            })

        # Create and save merged image
        if channels > 1:
            merged_images = []
            for result in results:
                with default_storage.open(result['image_url'].replace(settings.MEDIA_URL, '')) as f:
                    merged_images.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1))
            
            merged = merge_imgs(merged_images)
            merged = add_scale_bar(merged, (orix, oriy), swidth, 10)
            merged_with_border = add_border(merged, border_size=10, border_color=(255, 255, 255))
            sharpened_merged = sharpen_edges(merged_with_border)
            
            merged_file_name = f'{algorithm}_merged.png'
            merged_path = os.path.join(PROCESSED_DIR, merged_file_name)
            
            buffer = cv2.imencode('.png', sharpened_merged)[1].tobytes()
            merged_image_path = default_storage.save(merged_path, ContentFile(buffer))
            
            # Calculate accuracy for merged image
            accuracy = calculate_accuracy(original_img, sharpened_merged)

            results.append({
                'image_url': default_storage.url(merged_image_path),
                'algorithm': 'Merged Image with Edges',
                'accuracy': accuracy
            })

        return render(request, 'image_processing/result.html', {'results': results, 'selected_algorithm': algorithm})

    return JsonResponse({'error': 'Invalid request method!'}, status=405)


def index(request):
    return render(request, 'image_processing/upload.html')

import cv2
import os
import plotly.graph_objs as go
from django.shortcuts import render
from django.conf import settings

# def create_visualization(image_path, title):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image at path {image_path} could not be loaded.")
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # ... rest of your code

#     # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Generate a histogram for each color channel
#     fig = go.Figure()

#     for i, color in enumerate(['Red', 'Green', 'Blue']):
#         hist = cv2.calcHist([rgb_image], [i], None, [256], [0, 256])
#         fig.add_trace(go.Scatter(x=list(range(256)), y=hist.flatten(), mode='lines', name=color))
    
#     # Set layout for the figure
#     fig.update_layout(title=title, xaxis_title='Pixel Value', yaxis_title='Frequency', height=400)
    
#     return fig



from PIL import Image
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse
import os

# def convert_tif_to_png(tif_path, png_path):
#     """ Convert a .tif image to .png format. """
#     with Image.open(tif_path) as img:
#         img.save(png_path, format='PNG')


from sklearn.metrics import mean_squared_error
import numpy as np

from PIL import Image
import numpy as np

import cv2
import numpy as np
from PIL import Image

def normalize_clarity_score(score, min_score=0, max_score=10000):
    """Normalize the clarity score to a scale of 0 to 10."""
    if score < min_score:
        score = min_score
    elif score > max_score:
        score = max_score

    normalized_score = ((score - min_score) / (max_score - min_score)) * 10
    return round(normalized_score, 2)  # Round to 2 decimal places for readability


# image_processing/views.py
import os
import base64
import io
from django.shortcuts import render
from django.core.files.storage import default_storage
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

def calculate_image_clarity(image):
    """Calculate image clarity using variance of Laplacian."""
    image = np.array(image.convert("L"))  # Convert image to grayscale
    laplacian = cv2.Laplacian(image, cv2.CV_64F)  # Calculate the Laplacian
    clarity = laplacian.var()  # Calculate the variance of the Laplacian
    return clarity

# def normalize_clarity_score(clarity):
#     """Normalize clarity score for better comparison."""
#     return (clarity - 0) / (10000 - 0)  # Example normalization, adjust based on your data range

def detect_edges1(image, algorithm):
    """Apply edge detection to the image based on the selected algorithm."""
    image = np.array(image.convert("L"))
    if algorithm == 'canny':
        return cv2.Canny(image, 100, 200)
    elif algorithm == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    elif algorithm == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
    else:
        raise ValueError("Unknown edge detection algorithm")

def generate_clarity_chart():
    """Generate Plotly chart for normalized clarity scores."""
    algorithms = ['canny', 'sobel', 'laplacian']
    clarity_scores = []

    upload_dir = default_storage.path('uploaded_images')
    for algo in algorithms:
        uploaded_images = os.listdir(upload_dir)
        if uploaded_images:
            uploaded_image = uploaded_images[0]
            uploaded_image_path = os.path.join(upload_dir, uploaded_image)
            uploaded_image = Image.open(uploaded_image_path)
            
            processed_image = detect_edges1(uploaded_image, algo)
            processed_image_pil = Image.fromarray(processed_image)
            
            clarity = calculate_image_clarity(processed_image_pil)
            normalized_clarity = normalize_clarity_score(clarity)
            clarity_scores.append(normalized_clarity)
    
    fig = go.Figure(data=[go.Bar(
        x=algorithms,
        y=clarity_scores,
        text=[f"{score:.2f}" for score in clarity_scores],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Normalized Clarity Score by Algorithm',
        xaxis_title='Algorithm',
        yaxis_title='Normalized Clarity Score',
    )

    return fig.to_html(full_html=False)

import plotly.graph_objects as go

def generate_color_intensity_plot(image):
    """Generate a color intensity Plotly heatmap for the merged image."""
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the color intensity values (0 to 1)
    normalized_image = image_array / 255.0

    # Separate the color channels (if RGB)
    red_channel = normalized_image[:, :, 0]
    green_channel = normalized_image[:, :, 1]
    blue_channel = normalized_image[:, :, 2]

    # Calculate overall intensity (you can use different metrics here)
    intensity = 0.2989 * red_channel + 0.5870 * green_channel + 0.1140 * blue_channel  # Grayscale conversion

    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=intensity,
        colorscale='Viridis',
        zmin=0,
        zmax=1
    ))

    fig.update_layout(
        title='Color Intensity Heatmap',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
    )

    return fig.to_html(full_html=False)

import plotly.express as px

def generate_heatmap(image, color_channel='red'):
    """Generate a heatmap for the intensity values of a specific color channel."""
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Select the color channel (0=red, 1=green, 2=blue)
    color_channel_idx = {'red': 0, 'green': 1, 'blue': 2}[color_channel]
    color_channel_data = image_array[:, :, color_channel_idx]

    # Create a heatmap for the selected color channel
    fig = px.imshow(color_channel_data, color_continuous_scale='Viridis')

    fig.update_layout(
        title=f'{color_channel.capitalize()} Channel Intensity Heatmap',
        coloraxis_colorbar=dict(title="Intensity"),
    )

    return fig.to_html(full_html=False)

def generate_color_intensity_line_plot(image):
    """Generate a color intensity line plot for the merged image."""
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the color intensity values (0 to 1)
    normalized_image = image_array / 255.0

    # Separate the color channels (RGB)
    red_channel = normalized_image[:, :, 0]
    green_channel = normalized_image[:, :, 1]
    blue_channel = normalized_image[:, :, 2]

    # Calculate the mean intensity across rows (or columns)
    mean_red_intensity = red_channel.mean(axis=1)  # Mean intensity across rows for the red channel
    mean_green_intensity = green_channel.mean(axis=1)  # Mean intensity across rows for the green channel
    mean_blue_intensity = blue_channel.mean(axis=1)  # Mean intensity across rows for the blue channel

    # Create a line plot for each color channel
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=mean_red_intensity,
        mode='lines',
        line=dict(color='red'),
        name='Red Intensity'
    ))

    fig.add_trace(go.Scatter(
        y=mean_green_intensity,
        mode='lines',
        line=dict(color='green'),
        name='Green Intensity'
    ))

    fig.add_trace(go.Scatter(
        y=mean_blue_intensity,
        mode='lines',
        line=dict(color='blue'),
        name='Blue Intensity'
    ))

    fig.update_layout(
        title='Color Intensity Across Rows',
        xaxis_title='Row Index',
        yaxis_title='Mean Intensity',
        legend_title="Color Channels",
    )

    return fig.to_html(full_html=False)

import plotly.graph_objs as go
import numpy as np

def generate_contour_plot(image, color_channel='red'):
    image_array = np.array(image)
    color_channel_idx = {'red': 0, 'green': 1, 'blue': 2}[color_channel]
    color_channel_data = image_array[:, :, color_channel_idx]

    fig = go.Figure(data=go.Contour(
        z=color_channel_data,
        colorscale='Viridis',
        line_smoothing=0.85,
        contours=dict(coloring='heatmap')
    ))

    fig.update_layout(
        title=f'{color_channel.capitalize()} Channel Contour Plot',
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis')
    )

    return fig.to_html(full_html=False)

import plotly.graph_objs as go
import numpy as np
from PIL import Image

def get_color_channel_data(image_array, color_channel='red'):
    if len(image_array.shape) == 2:  # Grayscale image
        return image_array
    else:
        color_channel_idx = {'red': 0, 'green': 1, 'blue': 2}[color_channel]
        return image_array[:, :, color_channel_idx]

def generate_contour_plot(image, color_channel='red'):
    image_array = np.array(image)
    color_channel_data = get_color_channel_data(image_array, color_channel)

    fig = go.Figure(data=go.Contour(
        z=color_channel_data,
        colorscale='Viridis',
        line_smoothing=0.85,
        contours=dict(coloring='heatmap')
    ))

    fig.update_layout(
        title=f'{color_channel.capitalize()} Channel Contour Plot',
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis')
    )

    return fig.to_html(full_html=False)

import plotly.graph_objs as go
import numpy as np
from PIL import Image

def get_color_channel_data(image_array, color_channel='red'):
    if len(image_array.shape) == 2:  # Grayscale image
        return image_array
    else:
        color_channel_idx = {'red': 0, 'green': 1, 'blue': 2}[color_channel]
        return image_array[:, :, color_channel_idx]

def generate_3d_surface_plot(image, color_channel='red'):
    image_array = np.array(image)
    color_channel_data = get_color_channel_data(image_array, color_channel)

    # Normalize the data for better visualization
    z_data = color_channel_data / 255.0

    x_data, y_data = np.meshgrid(range(z_data.shape[1]), range(z_data.shape[0]))

    fig = go.Figure(data=[go.Surface(z=z_data, x=x_data, y=y_data, colorscale='Viridis')])

    fig.update_layout(
        title=f'3D Surface Plot - {color_channel.capitalize()} Channel',
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Intensity'
        )
    )

    return fig.to_html(full_html=False)


from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import JsonResponse
from PIL import Image
import os
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px

def convert_image_to_png(image_path):
    """Convert any image to .png format."""
    with Image.open(image_path) as img:
        png_path = os.path.splitext(image_path)[0] + '.png'
        img.save(png_path, format='PNG')
    return png_path

def report(request):
    if request.method == 'GET':
        upload_dir = default_storage.path('uploaded_images')
        processed_dir = default_storage.path('processed_images')

        # Ensure directories exist
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        # Get the uploaded image
        uploaded_images = os.listdir(upload_dir)
        if uploaded_images:
            uploaded_image = uploaded_images[0]
            uploaded_image_path = os.path.join(upload_dir, uploaded_image)

            if not uploaded_image.lower().endswith('.png'):
                try:
                    uploaded_image_path = convert_image_to_png(uploaded_image_path)
                except Exception as e:
                    return JsonResponse({'error': f'Error converting image: {str(e)}'}, status=500)

            uploaded_image_url = default_storage.url(os.path.join('uploaded_images', os.path.basename(uploaded_image_path)))
            uploaded_image = Image.open(uploaded_image_path)
        else:
            uploaded_image_url = None
            uploaded_image = None

        # Get the latest processed (merged) image
        processed_images = sorted(os.listdir(processed_dir), key=lambda x: os.path.getctime(os.path.join(processed_dir, x)))
        if processed_images:
            latest_processed_image = processed_images[-1]
            merged_image_path = os.path.join(processed_dir, latest_processed_image)
            merged_image_url = default_storage.url(os.path.join('processed_images', latest_processed_image))
            merged_image = Image.open(merged_image_path)
        else:
            merged_image_url = None
            merged_image = None

        # Calculate clarity if the images are available
        uploaded_image_clarity = calculate_image_clarity(uploaded_image) if uploaded_image else None
        merged_image_clarity = calculate_image_clarity(merged_image) if merged_image else None

        # Normalize clarity scores
        normalized_uploaded_image_clarity = normalize_clarity_score(uploaded_image_clarity) if uploaded_image_clarity else None
        normalized_merged_image_clarity = normalize_clarity_score(merged_image_clarity) if merged_image_clarity else None

        # Generate the Plotly charts
        clarity_chart_html = generate_clarity_chart()
        color_intensity_plot_html = generate_color_intensity_plot(merged_image) if merged_image else None
        color_intensity_line_plot_html = generate_color_intensity_line_plot(merged_image) if merged_image else None
        contour_plot_red = generate_contour_plot(merged_image, color_channel='red') if merged_image else None
        contour_plot_green = generate_contour_plot(merged_image, color_channel='green') if merged_image else None
        contour_plot_blue = generate_contour_plot(merged_image, color_channel='blue') if merged_image else None
        red_heatmap_html = generate_heatmap(merged_image, 'red') if merged_image else None
        green_heatmap_html = generate_heatmap(merged_image, 'green') if merged_image else None
        blue_heatmap_html = generate_heatmap(merged_image, 'blue') if merged_image else None
        surface_plot_red = generate_3d_surface_plot(uploaded_image, color_channel='red') if uploaded_image else None
        surface_plot_green = generate_3d_surface_plot(uploaded_image, color_channel='green') if uploaded_image else None
        surface_plot_blue = generate_3d_surface_plot(uploaded_image, color_channel='blue') if uploaded_image else None

        # Render the report.html template with the image URLs, clarity, and chart
        context = {
            'uploaded_image_url': uploaded_image_url,
            'merged_image_url': merged_image_url,
            'normalized_uploaded_image_clarity': normalized_uploaded_image_clarity,
            'normalized_merged_image_clarity': normalized_merged_image_clarity,
            'clarity_chart_html': clarity_chart_html,
            'color_intensity_plot_html': color_intensity_plot_html,
            'color_intensity_line_plot_html': color_intensity_line_plot_html,
            'red_heatmap_html': red_heatmap_html,
            'green_heatmap_html': green_heatmap_html,
            'blue_heatmap_html': blue_heatmap_html,
            'contour_plot_red': contour_plot_red,
            'contour_plot_green': contour_plot_green,
            'contour_plot_blue': contour_plot_blue,
            'surface_plot_red': surface_plot_red,
            'surface_plot_green': surface_plot_green,
            'surface_plot_blue': surface_plot_blue,
        }
        return render(request, 'image_processing/report.html', context)

    return JsonResponse({'error': 'Invalid request method!'}, status=405)
