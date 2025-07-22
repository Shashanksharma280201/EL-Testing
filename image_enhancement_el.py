import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from skimage import filters, morphology, exposure, restoration, measure
from scipy import ndimage
from sklearn.cluster import KMeans

def denoise_and_sharpen(image_path, output_path=None):
    """
    Advanced denoising and sharpening for solar panel images
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Advanced denoising using Non-local Means
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: Enhance contrast using CLAHE with optimal parameters
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # Step 3: Morphological cleaning to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Step 4: Advanced sharpening using unsharp mask
    gaussian_blur = cv2.GaussianBlur(cleaned, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(cleaned, 2.0, gaussian_blur, -1.0, 0)
    
    # Step 5: Edge-preserving filter
    edge_preserved = cv2.edgePreservingFilter(unsharp_mask, flags=2, sigma_s=50, sigma_r=0.4)
    
    # Step 6: Final sharpening
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                              [-1, 2, 2, 2,-1],
                              [-1, 2, 8, 2,-1],
                              [-1, 2, 2, 2,-1],
                              [-1,-1,-1,-1,-1]]) / 8.0
    
    sharpened = cv2.filter2D(edge_preserved, -1, kernel_sharpen)
    
    # Ensure values are in valid range
    result = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

def advanced_solar_enhancement(image_path, output_path=None):
    """
    Ultra-clean enhancement specifically for solar panels
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Bilateral filter for edge-preserving smoothing
    bilateral = cv2.bilateralFilter(gray, 15, 80, 80)
    
    # Step 2: Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(bilateral, None, h=8, templateWindowSize=7, searchWindowSize=21)
    
    # Step 3: Morphological gradient to enhance edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morph_grad = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
    
    # Step 4: Combine original with morphological gradient
    enhanced = cv2.addWeighted(denoised, 0.85, morph_grad, 0.15, 0)
    
    # Step 5: Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    equalized = clahe.apply(enhanced)
    
    # Step 6: Gaussian high-pass filter for sharpening
    blur = cv2.GaussianBlur(equalized, (0, 0), 1.5)
    high_pass = cv2.addWeighted(equalized, 1.5, blur, -0.5, 0)
    
    # Step 7: Remove remaining noise with median filter
    median_filtered = cv2.medianBlur(high_pass, 3)
    
    # Step 8: Final edge enhancement
    laplacian = cv2.Laplacian(median_filtered, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.absolute(laplacian))
    result = cv2.addWeighted(median_filtered, 0.9, laplacian, 0.1, 0)
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

def professional_solar_processing(image_path, output_path=None):
    """
    Professional-grade processing for solar panel inspection
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Multiple-stage denoising
    # Stage 1: Bilateral filter
    stage1 = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Stage 2: Non-local means
    stage2 = cv2.fastNlMeansDenoising(stage1, None, h=12, templateWindowSize=7, searchWindowSize=21)
    
    # Stage 3: Morphological opening to remove small artifacts
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    stage3 = cv2.morphologyEx(stage2, cv2.MORPH_OPEN, kernel_small)
    
    # Step 2: Enhance panel structure
    # Create kernels for detecting panel grid lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    # Detect horizontal and vertical structures
    horizontal = cv2.morphologyEx(stage3, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(stage3, cv2.MORPH_OPEN, kernel_v)
    
    # Combine structures
    structure = cv2.add(horizontal, vertical)
    
    # Enhance the original with detected structure
    structured = cv2.addWeighted(stage3, 0.8, structure, 0.2, 0)
    
    # Step 3: Contrast enhancement with gamma correction
    # Normalize to 0-1 range
    normalized = structured.astype(np.float32) / 255.0
    
    # Apply gamma correction
    gamma_corrected = np.power(normalized, 0.7)
    
    # Apply adaptive histogram equalization
    gamma_uint8 = (gamma_corrected * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gamma_uint8)
    
    # Step 4: Advanced sharpening
    # Create a strong sharpening kernel
    kernel_sharpen = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
    
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel_sharpen)
    
    # Step 5: Edge enhancement using Sobel
    sobelx = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    sobel_normalized = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    # Combine with original
    edge_enhanced = cv2.addWeighted(sharpened, 0.85, sobel_normalized, 0.15, 0)
    
    # Step 6: Final cleanup
    # Remove any remaining noise while preserving edges
    final = cv2.bilateralFilter(edge_enhanced, 5, 50, 50)
    
    if output_path:
        cv2.imwrite(output_path, final)
    
    return final

def defect_optimized_enhancement(image_path, output_path=None):
    """
    Specifically optimized for crack and defect detection
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Aggressive noise reduction
    # Multiple passes of different filters
    denoised1 = cv2.medianBlur(gray, 5)  # Remove salt-and-pepper noise
    denoised2 = cv2.GaussianBlur(denoised1, (3, 3), 0)  # Smooth Gaussian noise
    denoised3 = cv2.bilateralFilter(denoised2, 9, 75, 75)  # Edge-preserving
    
    # Non-local means for final denoising
    final_denoised = cv2.fastNlMeansDenoising(denoised3, None, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_contrast = clahe.apply(final_denoised)
    
    # Step 3: Morphological operations to clean up
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    cleaned = cv2.morphologyEx(enhanced_contrast, cv2.MORPH_CLOSE, kernel_clean)
    
    # Step 4: Detect and enhance linear features (cracks)
    # Create directional kernels
    kernel_45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    kernel_135 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    
    # Apply morphological operations
    lines_h = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_h)
    lines_v = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_v)
    lines_45 = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_45)
    lines_135 = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_135)
    
    # Combine all directional features
    all_lines = cv2.add(cv2.add(lines_h, lines_v), cv2.add(lines_45, lines_135))
    
    # Enhance original with detected lines
    line_enhanced = cv2.addWeighted(cleaned, 0.8, all_lines, 0.2, 0)
    
    # Step 5: Multi-scale sharpening
    # Fine details
    fine_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    fine_sharpened = cv2.filter2D(line_enhanced, -1, fine_kernel)
    
    # Medium details
    medium_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    medium_sharpened = cv2.filter2D(fine_sharpened, -1, medium_kernel)
    
    # Combine sharpening levels
    multi_sharpened = cv2.addWeighted(fine_sharpened, 0.6, medium_sharpened, 0.4, 0)
    
    # Step 6: Final edge enhancement
    edges = cv2.Canny(multi_sharpened, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    # Combine with original
    result = cv2.addWeighted(multi_sharpened, 0.9, edges_dilated, 0.1, 0)
    
    # Ensure proper range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

def compare_enhanced_methods(image_path):
    """
    Compare all enhanced methods
    """
    # Load original
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply all methods
    method1 = denoise_and_sharpen(image_path)
    method2 = advanced_solar_enhancement(image_path)
    method3 = professional_solar_processing(image_path)
    method4 = defect_optimized_enhancement(image_path)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Image', fontsize=12)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(method1, cmap='gray')
    axes[0,1].set_title('Method 1: Denoise & Sharpen', fontsize=12)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(method2, cmap='gray')
    axes[0,2].set_title('Method 2: Advanced Solar', fontsize=12)
    axes[0,2].axis('off')
    
    axes[1,0].imshow(method3, cmap='gray')
    axes[1,0].set_title('Method 3: Professional Processing', fontsize=12)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(method4, cmap='gray')
    axes[1,1].set_title('Method 4: Defect Optimized', fontsize=12)
    axes[1,1].axis('off')
    
    # Calculate and show image quality metrics
    axes[1,2].axis('off')
    
    # Calculate metrics
    def calculate_sharpness(img):
        return cv2.Laplacian(img, cv2.CV_64F).var()
    
    def calculate_contrast(img):
        return img.std()
    
    metrics_text = f"""Image Quality Metrics:
    
Original:
Sharpness: {calculate_sharpness(original):.1f}
Contrast: {calculate_contrast(original):.1f}

Method 1:
Sharpness: {calculate_sharpness(method1):.1f}
Contrast: {calculate_contrast(method1):.1f}

Method 2:
Sharpness: {calculate_sharpness(method2):.1f}
Contrast: {calculate_contrast(method2):.1f}

Method 3:
Sharpness: {calculate_sharpness(method3):.1f}
Contrast: {calculate_contrast(method3):.1f}

Method 4:
Sharpness: {calculate_sharpness(method4):.1f}
Contrast: {calculate_contrast(method4):.1f}
    """
    
    axes[1,2].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('enhanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return method1, method2, method3, method4

# Ultra-clean processing pipeline
def ultra_clean_pipeline(image_path, output_path=None):
    """
    Ultra-clean processing pipeline combining best techniques
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Stage 1: Noise removal
    # Multiple denoising stages
    stage1 = cv2.medianBlur(gray, 3)  # Remove impulse noise
    stage2 = cv2.GaussianBlur(stage1, (3, 3), 0.5)  # Smooth noise
    stage3 = cv2.bilateralFilter(stage2, 9, 75, 75)  # Edge-preserving
    stage4 = cv2.fastNlMeansDenoising(stage3, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Stage 2: Structure enhancement
    # Enhance panel grid structure
    kernel_rect_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    kernel_rect_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    
    horizontal_lines = cv2.morphologyEx(stage4, cv2.MORPH_OPEN, kernel_rect_h)
    vertical_lines = cv2.morphologyEx(stage4, cv2.MORPH_OPEN, kernel_rect_v)
    grid_structure = cv2.add(horizontal_lines, vertical_lines)
    
    # Combine with original
    structure_enhanced = cv2.addWeighted(stage4, 0.85, grid_structure, 0.15, 0)
    
    # Stage 3: Contrast and brightness optimization
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_optimized = clahe.apply(structure_enhanced)
    
    # Gamma correction for better visibility
    gamma = 0.8
    gamma_corrected = np.power(contrast_optimized.astype(np.float32) / 255.0, gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    
    # Stage 4: Precision sharpening
    # Create precision sharpening kernel
    precision_kernel = np.array([[0, -0.5, 0],
                                [-0.5, 3, -0.5],
                                [0, -0.5, 0]])
    
    precision_sharpened = cv2.filter2D(gamma_corrected, -1, precision_kernel)
    
    # Stage 5: Edge enhancement
    # Calculate gradients
    grad_x = cv2.Sobel(precision_sharpened, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(precision_sharpened, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_normalized = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    # Combine with sharpened image
    edge_enhanced = cv2.addWeighted(precision_sharpened, 0.9, gradient_normalized, 0.1, 0)
    
    # Stage 6: Final cleanup
    # Very light bilateral filter to remove any remaining artifacts
    final_result = cv2.bilateralFilter(edge_enhanced, 3, 25, 25)
    
    if output_path:
        cv2.imwrite(output_path, final_result)
    
    return final_result

# Example usage
if __name__ == "__main__":
    input_image = "/home/flo/Videos/EL project/EL_Test/SH151009P636KSPC-521.jpg"
    
    try:
        print("Processing with enhanced methods...")
        
        # Method 1: Denoise and sharpen
        result1 = denoise_and_sharpen(input_image, "enhanced_denoised.jpg")
        print("Method 1 completed: Denoise & Sharpen")
        
        # Method 2: Advanced solar enhancement
        result2 = advanced_solar_enhancement(input_image, "enhanced_advanced_solar.jpg")
        print("Method 2 completed: Advanced Solar")
        
        # Method 3: Professional processing
        result3 = professional_solar_processing(input_image, "enhanced_professional.jpg")
        print("Method 3 completed: Professional Processing")
        
        # Method 4: Defect optimized
        result4 = defect_optimized_enhancement(input_image, "enhanced_defect_optimized.jpg")
        print("Method 4 completed: Defect Optimized")
        
        # Ultra-clean pipeline
        result5 = ultra_clean_pipeline(input_image, "enhanced_ultra_clean.jpg")
        print("Ultra-clean pipeline completed")
        
        # Compare all methods
        compare_enhanced_methods(input_image)
        print("Comparison completed and saved")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check if the image file exists")

# Batch processing with best method
def batch_process_ultra_clean(input_folder, output_folder):
    """
    Batch process with ultra-clean pipeline
    """
    import os
    from pathlib import Path
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    print(f"Processing {len(image_files)} images with ultra-clean pipeline...")
    
    for i, img_path in enumerate(image_files):
        try:
            output_path = Path(output_folder) / f"ultra_clean_{img_path.name}"
            ultra_clean_pipeline(str(img_path), str(output_path))
            print(f"Processed {i+1}/{len(image_files)}: {img_path.name}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    print("Batch processing completed!")