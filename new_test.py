import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def detect_solar_panel_area(gray_image):
    """
    Detect the main solar panel area and remove black borders
    """
    # Apply initial denoising for better detection
    denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    # Use adaptive threshold for better results with noisy images
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Also try regular threshold
    _, binary_thresh = cv2.threshold(denoised, 30, 255, cv2.THRESH_BINARY)
    
    # Combine both thresholding methods
    combined_thresh = cv2.bitwise_or(adaptive_thresh, binary_thresh)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
    cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours to get panel boundary
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (should be the panel)
        largest_contour = max(contours, key=cv2.contourArea)
        panel_x, panel_y, panel_w, panel_h = cv2.boundingRect(largest_contour)
        
        # Add some padding to ensure we don't cut off cells
        padding = 5
        panel_x = max(0, panel_x - padding)
        panel_y = max(0, panel_y - padding)
        panel_w = min(gray_image.shape[1] - panel_x, panel_w + 2*padding)
        panel_h = min(gray_image.shape[0] - panel_y, panel_h + 2*padding)
        
        return panel_x, panel_y, panel_w, panel_h, denoised
    else:
        # Fallback: use entire image
        return 0, 0, gray_image.shape[1], gray_image.shape[0], denoised

def generate_cell_grid(panel_x, panel_y, panel_w, panel_h, grid_rows=6, grid_cols=10):
    """
    Generate grid coordinates for individual solar cells
    """
    cell_width = panel_w // grid_cols
    cell_height = panel_h // grid_rows
    
    cell_boxes = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = panel_x + col * cell_width
            y = panel_y + row * cell_height
            
            # Add small overlap to ensure we don't miss edges
            x_start = max(0, x - 2)
            y_start = max(0, y - 2)
            w = cell_width + 4
            h = cell_height + 4
            
            cell_boxes.append((x_start, y_start, w, h, row, col))
    
    return cell_boxes

def sharp_clean_enhancement(cell_image, target_size=512):
    """
    Sharp and clean enhancement with minimal blur for crisp results
    """
    if cell_image is None or cell_image.size == 0:
        return None
    
    h, w = cell_image.shape
    if h == 0 or w == 0:
        return None
    
    # Step 1: High-quality upscaling
    scale_factor = target_size / max(h, w)
    current_img = cell_image.copy()
    current_scale = 1.0
    
    while current_scale < scale_factor:
        next_scale = min(2.0, scale_factor / current_scale)
        current_h, current_w = current_img.shape
        new_h, new_w = int(current_h * next_scale), int(current_w * next_scale)
        current_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        current_scale *= next_scale
    
    upscaled = current_img
    
    # Make square by padding
    if upscaled.shape[0] != upscaled.shape[1]:
        max_dim = max(upscaled.shape)
        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        start_y = (max_dim - upscaled.shape[0]) // 2
        start_x = (max_dim - upscaled.shape[1]) // 2
        padded[start_y:start_y+upscaled.shape[0], start_x:start_x+upscaled.shape[1]] = upscaled
        upscaled = padded
    
    print(f"    Applying sharp enhancement to {upscaled.shape[0]}x{upscaled.shape[1]} cell...")
    
    # Step 2: Minimal noise reduction (preserve detail)
    # Only light median filter for severe noise
    stage1 = cv2.medianBlur(upscaled, 3)
    
    # Single light bilateral filter
    denoised = cv2.bilateralFilter(stage1, 7, 50, 50)
    
    # Step 3: Aggressive contrast enhancement
    # Strong histogram stretching
    p0_2, p99_8 = np.percentile(denoised, (0.2, 99.8))
    if p99_8 > p0_2:
        stretched = np.clip((denoised - p0_2) * 255 / (p99_8 - p0_2), 0, 255).astype(np.uint8)
    else:
        stretched = denoised
    
    # Strong CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(stretched)
    
    # Step 4: Multi-stage sharpening
    # Stage 1: Strong unsharp masking
    blur_small = cv2.GaussianBlur(contrast_enhanced, (0, 0), 0.8)
    unsharp1 = cv2.addWeighted(contrast_enhanced, 2.2, blur_small, -1.2, 0)
    
    # Stage 2: Detail enhancement
    blur_detail = cv2.GaussianBlur(unsharp1, (0, 0), 0.3)
    detail_enhanced = cv2.addWeighted(unsharp1, 1.6, blur_detail, -0.6, 0)
    
    # Step 5: Strong edge enhancement
    # Multiple edge detection methods
    sobelx = cv2.Sobel(detail_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(detail_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Laplacian for additional edge detection
    laplacian = cv2.Laplacian(detail_enhanced, cv2.CV_64F, ksize=3)
    laplacian = np.absolute(laplacian)
    
    # Combine edge enhancements
    if sobel_magnitude.max() > 0 and laplacian.max() > 0:
        sobel_norm = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)
        laplacian_norm = (laplacian / laplacian.max() * 255).astype(np.uint8)
        
        # Combine both edge methods
        combined_edges = cv2.addWeighted(sobel_norm, 0.6, laplacian_norm, 0.4, 0)
        edge_enhanced = cv2.addWeighted(detail_enhanced, 0.8, combined_edges, 0.2, 0)
    else:
        edge_enhanced = detail_enhanced
    
    # Step 6: Convolution sharpening
    # Create strong sharpening kernel
    kernel_sharp = np.array([[-0.5, -1, -0.5],
                            [-1,   7, -1],
                            [-0.5, -1, -0.5]])
    conv_sharpened = cv2.filter2D(edge_enhanced, -1, kernel_sharp)
    
    # Blend sharpening
    sharpened = cv2.addWeighted(edge_enhanced, 0.6, conv_sharpened, 0.4, 0)
    
    # Step 7: Final contrast and gamma adjustment
    # Gamma correction for punch
    gamma = 0.8
    gamma_corrected = np.power(sharpened.astype(np.float32) / 255.0, gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    
    # Final contrast boost
    final_contrast = cv2.convertScaleAbs(gamma_corrected, alpha=1.1, beta=5)
    
    # Ensure proper range
    final_result = np.clip(final_contrast, 0, 255).astype(np.uint8)
    
    return final_result

def ultra_clean_enhancement(cell_image, target_size=512):
    """
    Ultra-clean enhancement with reduced blur and aggressive sharpening
    """
    if cell_image is None or cell_image.size == 0:
        return None
    
    h, w = cell_image.shape
    if h == 0 or w == 0:
        return None
    
    # Step 1: Progressive high-quality upscaling
    scale_factor = target_size / max(h, w)
    current_img = cell_image.copy()
    current_scale = 1.0
    
    # Progressive scaling to avoid artifacts
    while current_scale < scale_factor:
        next_scale = min(2.0, scale_factor / current_scale)
        current_h, current_w = current_img.shape
        new_h, new_w = int(current_h * next_scale), int(current_w * next_scale)
        
        # Use highest quality interpolation
        current_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        current_scale *= next_scale
    
    upscaled = current_img
    
    # Make square by padding if needed
    if upscaled.shape[0] != upscaled.shape[1]:
        max_dim = max(upscaled.shape)
        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        start_y = (max_dim - upscaled.shape[0]) // 2
        start_x = (max_dim - upscaled.shape[1]) // 2
        padded[start_y:start_y+upscaled.shape[0], start_x:start_x+upscaled.shape[1]] = upscaled
        upscaled = padded
    
    # Step 2: Reduced blur noise reduction (preserve more detail)
    print(f"    Applying optimized enhancement to {upscaled.shape[0]}x{upscaled.shape[1]} cell...")
    
    # Stage 1: Light median filter only for impulse noise
    stage1 = cv2.medianBlur(upscaled, 3)  # Reduced from 5
    
    # Stage 2: Skip Gaussian blur to preserve sharpness
    
    # Stage 3: Reduced bilateral filtering (preserve edges better)
    stage2 = cv2.bilateralFilter(stage1, 9, 60, 60)  # Reduced parameters
    stage3 = cv2.bilateralFilter(stage2, 5, 40, 40)  # Lighter second pass
    
    # Stage 4: Lighter non-local means denoising
    stage4 = cv2.fastNlMeansDenoising(stage3, None, h=12, templateWindowSize=7, searchWindowSize=15)  # Reduced h and search window
    
    # Stage 5: Skip morphological operations that can blur details
    
    # Step 3: Enhanced contrast enhancement
    # More aggressive histogram stretching
    p0_5, p99_5 = np.percentile(stage4, (0.5, 99.5))
    if p99_5 > p0_5:
        stretched = np.clip((stage4 - p0_5) * 255 / (p99_5 - p0_5), 0, 255).astype(np.uint8)
    else:
        stretched = stage4
    
    # More aggressive CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))  # Increased clip limit
    contrast_enhanced = clahe.apply(stretched)
    
    # Step 4: Multi-level sharpening
    # Level 1: Unsharp masking with smaller blur radius
    blur_unsharp = cv2.GaussianBlur(contrast_enhanced, (0, 0), 1.0)  # Reduced sigma
    unsharp_mask = cv2.addWeighted(contrast_enhanced, 1.8, blur_unsharp, -0.8, 0)  # More aggressive
    
    # Level 2: High-frequency detail enhancement
    blur_detail = cv2.GaussianBlur(unsharp_mask, (0, 0), 0.5)  # Very small blur
    detail_enhanced = cv2.addWeighted(unsharp_mask, 1.4, blur_detail, -0.4, 0)
    
    # Step 5: Strong edge enhancement
    # Use smaller kernel Sobel for sharper edges
    sobelx = cv2.Sobel(detail_enhanced, cv2.CV_64F, 1, 0, ksize=3)  # Reduced kernel size
    sobely = cv2.Sobel(detail_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Apply more aggressively
    if sobel_magnitude.max() > 0:
        sobel_normalized = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)
        edge_enhanced = cv2.addWeighted(detail_enhanced, 0.85, sobel_normalized, 0.15, 0)  # Increased edge contribution
    else:
        edge_enhanced = detail_enhanced
    
    # Step 6: Additional sharpening with convolution kernel
    # Create a strong sharpening kernel
    kernel_sharpen = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
    kernel_sharpened = cv2.filter2D(edge_enhanced, -1, kernel_sharpen)
    
    # Blend with edge enhanced version
    sharpened = cv2.addWeighted(edge_enhanced, 0.7, kernel_sharpened, 0.3, 0)
    
    # Step 7: Final contrast boost
    # Gamma correction for better visibility
    gamma = 0.85  # Slightly lower gamma for more contrast
    gamma_corrected = np.power(sharpened.astype(np.float32) / 255.0, gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    
    # Step 8: Very light final cleanup (minimal blur)
    final_result = cv2.bilateralFilter(gamma_corrected, 3, 15, 15)  # Much lighter final filter
    
    # Ensure proper value range
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    
    return final_result

def extract_and_enhance_all_cells(image_path, output_folder, target_size=512, grid_rows=6, grid_cols=10, enhancement_mode='sharp'):
    """
    Extract and enhance all individual solar cells with different enhancement modes
    
    enhancement_mode options:
    - 'sharp': Maximum sharpness with minimal blur
    - 'balanced': Balance between noise reduction and sharpness
    - 'clean': Maximum noise reduction (may be softer)
    """
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Original image size: {gray.shape[1]}x{gray.shape[0]}")
    
    # Detect solar panel area
    print("Detecting solar panel area...")
    panel_x, panel_y, panel_w, panel_h, preprocessed = detect_solar_panel_area(gray)
    print(f"Panel area detected: {panel_w}x{panel_h} at ({panel_x}, {panel_y})")
    
    # Generate cell grid
    print(f"Generating {grid_rows}x{grid_cols} cell grid...")
    cell_boxes = generate_cell_grid(panel_x, panel_y, panel_w, panel_h, grid_rows, grid_cols)
    print(f"Generated {len(cell_boxes)} cell coordinates")
    
    # Process each cell
    enhanced_cells = []
    successful_cells = 0
    
    print(f"\nProcessing individual cells with '{enhancement_mode}' mode...")
    print(f"Target size per cell: {target_size}x{target_size} pixels")
    print("-" * 50)
    
    for i, (x, y, w, h, row, col) in enumerate(cell_boxes):
        try:
            # Extract cell from preprocessed image
            cell = preprocessed[y:y+h, x:x+w]
            
            # Skip if cell is too small or mostly black
            if cell.size == 0 or np.mean(cell) < 15:
                print(f"Cell {i+1:2d} (Row {row+1}, Col {col+1}): Skipped - too dark/small")
                continue
            
            # Choose enhancement method
            print(f"Cell {i+1:2d} (Row {row+1}, Col {col+1}): Processing {cell.shape[1]}x{cell.shape[0]} → {target_size}x{target_size}")
            
            if enhancement_mode == 'sharp':
                enhanced_cell = sharp_clean_enhancement(cell, target_size)
            elif enhancement_mode == 'balanced':
                enhanced_cell = ultra_clean_enhancement(cell, target_size)
            else:  # 'clean'
                enhanced_cell = ultra_clean_enhancement(cell, target_size)
            
            if enhanced_cell is not None:
                # Save enhanced cell
                filename = f"cell_r{row+1:02d}_c{col+1:02d}_{i+1:03d}_{enhancement_mode}.jpg"
                output_path = Path(output_folder) / filename
                
                # Save with maximum quality for sharp images
                cv2.imwrite(str(output_path), enhanced_cell, [cv2.IMWRITE_JPEG_QUALITY, 98])
                
                enhanced_cells.append({
                    'image': enhanced_cell,
                    'row': row,
                    'col': col,
                    'index': i,
                    'filename': filename
                })
                successful_cells += 1
                print(f"    ✓ Saved: {filename}")
            else:
                print(f"    ✗ Enhancement failed")
        
        except Exception as e:
            print(f"    ✗ Error processing cell {i+1}: {e}")
            continue
    
    print("-" * 50)
    print(f"✅ Successfully processed {successful_cells}/{len(cell_boxes)} cells")
    print(f"📁 Enhanced cells saved to: {output_folder}")
    
    return enhanced_cells

def create_cells_overview(enhanced_cells, output_folder, grid_rows=6, grid_cols=10):
    """
    Create an overview image showing all enhanced cells
    """
    if not enhanced_cells:
        print("No enhanced cells to create overview")
        return
    
    print(f"\nCreating overview of {len(enhanced_cells)} enhanced cells...")
    
    # Limit display size for overview
    display_rows = min(4, grid_rows)
    display_cols = min(6, grid_cols)
    
    fig, axes = plt.subplots(display_rows, display_cols, figsize=(20, 14))
    
    # Handle single row/column cases
    if display_rows == 1 and display_cols == 1:
        axes = np.array([[axes]])
    elif display_rows == 1:
        axes = axes.reshape(1, -1)
    elif display_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Display cells
    cells_displayed = 0
    for row in range(display_rows):
        for col in range(display_cols):
            cell_index = row * grid_cols + col
            
            if cell_index < len(enhanced_cells):
                cell_data = enhanced_cells[cell_index]
                axes[row, col].imshow(cell_data['image'], cmap='gray')
                axes[row, col].set_title(
                    f"Cell {cell_data['index']+1}\n(R{cell_data['row']+1}, C{cell_data['col']+1})", 
                    fontsize=10
                )
                cells_displayed += 1
            else:
                axes[row, col].set_title('No Cell', fontsize=10)
                axes[row, col].set_facecolor('black')
            
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    plt.suptitle(f'Enhanced Solar Cells Overview - {cells_displayed} cells shown', fontsize=16)
    plt.tight_layout()
    
    # Save overview
    overview_path = Path(output_folder) / "enhanced_cells_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Overview saved: {overview_path}")

def analyze_cell_quality(enhanced_cells):
    """
    Analyze quality metrics of enhanced cells
    """
    if not enhanced_cells:
        return
    
    print(f"\n📈 Quality Analysis of {len(enhanced_cells)} Enhanced Cells:")
    print("=" * 80)
    print(f"{'Cell':<6} {'Row':<4} {'Col':<4} {'Sharpness':<10} {'Contrast':<9} {'Brightness':<10} {'Quality':<8}")
    print("-" * 80)
    
    sharpness_scores = []
    contrast_scores = []
    brightness_scores = []
    
    for cell_data in enhanced_cells:
        cell_img = cell_data['image']
        
        # Calculate metrics
        sharpness = cv2.Laplacian(cell_img, cv2.CV_64F).var()
        contrast = cell_img.std()
        brightness = cell_img.mean()
        
        # Simple quality score (higher = better)
        quality_score = (sharpness/1000 + contrast/50) * (brightness/128)
        
        sharpness_scores.append(sharpness)
        contrast_scores.append(contrast)
        brightness_scores.append(brightness)
        
        # Determine quality rating
        if quality_score > 2.0:
            quality_rating = "Excellent"
        elif quality_score > 1.5:
            quality_rating = "Good"
        elif quality_score > 1.0:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        print(f"{cell_data['index']+1:<6} {cell_data['row']+1:<4} {cell_data['col']+1:<4} "
              f"{sharpness:<10.1f} {contrast:<9.1f} {brightness:<10.1f} {quality_rating:<8}")
    
    # Summary statistics
    print("-" * 80)
    print(f"📊 Summary Statistics:")
    print(f"   Average Sharpness: {np.mean(sharpness_scores):.1f} (±{np.std(sharpness_scores):.1f})")
    print(f"   Average Contrast:  {np.mean(contrast_scores):.1f} (±{np.std(contrast_scores):.1f})")
    print(f"   Average Brightness: {np.mean(brightness_scores):.1f} (±{np.std(brightness_scores):.1f})")

def quick_single_cell_test(image_path, cell_index=10, target_size=1024, enhancement_mode='sharp'):
    """
    Quick test on a single cell with different enhancement modes comparison
    """
    print(f"🧪 Quick test: Enhancing cell {cell_index} at {target_size}x{target_size} with '{enhancement_mode}' mode")
    
    try:
        # Load and preprocess
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        panel_x, panel_y, panel_w, panel_h, preprocessed = detect_solar_panel_area(gray)
        cell_boxes = generate_cell_grid(panel_x, panel_y, panel_w, panel_h)
        
        if cell_index >= len(cell_boxes):
            print(f"❌ Cell index {cell_index} out of range (max: {len(cell_boxes)-1})")
            return None
        
        # Extract cell
        x, y, w, h, row, col = cell_boxes[cell_index]
        original_cell = gray[y:y+h, x:x+w]  # Use original for comparison
        preprocessed_cell = preprocessed[y:y+h, x:x+w]
        
        # Test different enhancement modes
        enhanced_balanced = ultra_clean_enhancement(preprocessed_cell, target_size)
        enhanced_sharp = sharp_clean_enhancement(preprocessed_cell, target_size)
        
        if enhanced_balanced is None or enhanced_sharp is None:
            print("❌ Enhancement failed")
            return None
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Original (noisy)
        axes[0,0].imshow(original_cell, cmap='gray')
        axes[0,0].set_title(f'Original Cell {cell_index}\n(Row {row+1}, Col {col+1})\nSize: {original_cell.shape[1]}x{original_cell.shape[0]}\nNoisy & Grainy', fontsize=12)
        axes[0,0].axis('off')
        
        # Preprocessed (denoised)
        axes[0,1].imshow(preprocessed_cell, cmap='gray')
        axes[0,1].set_title(f'Preprocessed Cell\nBasic Noise Reduction\nSize: {preprocessed_cell.shape[1]}x{preprocessed_cell.shape[0]}', fontsize=12)
        axes[0,1].axis('off')
        
        # Balanced enhancement
        axes[1,0].imshow(enhanced_balanced, cmap='gray')
        axes[1,0].set_title(f'Balanced Enhancement\nNoise Reduced + Moderate Sharp\nSize: {enhanced_balanced.shape[1]}x{enhanced_balanced.shape[0]}', fontsize=12)
        axes[1,0].axis('off')
        
        # Sharp enhancement
        axes[1,1].imshow(enhanced_sharp, cmap='gray')
        axes[1,1].set_title(f'Sharp Enhancement\nMaximum Sharpness\nSize: {enhanced_sharp.shape[1]}x{enhanced_sharp.shape[0]}', fontsize=12)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison and enhanced cells
        comparison_path = f'cell_{cell_index}_sharp_vs_balanced_comparison.png'
        balanced_path = f'cell_{cell_index}_balanced.jpg'
        sharp_path = f'cell_{cell_index}_sharp.jpg'
        
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        cv2.imwrite(balanced_path, enhanced_balanced, [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(sharp_path, enhanced_sharp, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        plt.show()
        
        # Calculate improvement metrics
        def calculate_sharpness(img):
            return cv2.Laplacian(img, cv2.CV_64F).var()
        
        def calculate_contrast(img):
            return img.std()
        
        original_sharpness = calculate_sharpness(original_cell)
        balanced_sharpness = calculate_sharpness(enhanced_balanced)
        sharp_sharpness = calculate_sharpness(enhanced_sharp)
        
        original_contrast = calculate_contrast(original_cell)
        balanced_contrast = calculate_contrast(enhanced_balanced)
        sharp_contrast = calculate_contrast(enhanced_sharp)
        
        print(f"📊 Enhancement Comparison Results:")
        print(f"   Original size:      {original_cell.shape[1]}x{original_cell.shape[0]}")
        print(f"   Enhanced size:      {enhanced_sharp.shape[1]}x{enhanced_sharp.shape[0]}")
        print(f"   Size increase:      {enhanced_sharp.shape[0] // original_cell.shape[0]}x")
        print(f"")
        print(f"   Sharpness Metrics:")
        print(f"     Original:         {original_sharpness:.1f}")
        print(f"     Balanced:         {balanced_sharpness:.1f} ({(balanced_sharpness/original_sharpness-1)*100:+.1f}%)")
        print(f"     Sharp:            {sharp_contrast:.1f} ({(sharp_contrast/original_contrast-1)*100:+.1f}%)")
        print(f"")
        print(f"   Files saved:")
        print(f"     - Comparison:     {comparison_path}")
        print(f"     - Balanced:       {balanced_path}")
        print(f"     - Sharp:          {sharp_path}")
        
        # Return the chosen enhancement mode
        return enhanced_sharp if enhancement_mode == 'sharp' else enhanced_balanced
        
    except Exception as e:
        print(f"❌ Error in quick test: {e}")
        return None

def enhance_specific_cells(image_path, output_folder, cell_indices, target_size=1024, grid_rows=6, grid_cols=10, enhancement_mode='sharp'):
    """
    Enhance only specific cells at higher resolution
    """
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    panel_x, panel_y, panel_w, panel_h, preprocessed = detect_solar_panel_area(gray)
    cell_boxes = generate_cell_grid(panel_x, panel_y, panel_w, panel_h, grid_rows, grid_cols)
    
    print(f"Enhancing specific cells at {target_size}x{target_size} resolution with '{enhancement_mode}' mode...")
    
    enhanced_cells = []
    for idx in cell_indices:
        if idx >= len(cell_boxes):
            print(f"Cell index {idx} out of range (max: {len(cell_boxes)-1})")
            continue
        
        x, y, w, h, row, col = cell_boxes[idx]
        
        try:
            # Extract cell
            cell = preprocessed[y:y+h, x:x+w]
            
            if cell.size == 0 or np.mean(cell) < 15:
                print(f"Cell {idx} (Row {row+1}, Col {col+1}): Skipped - too dark")
                continue
            
            # Enhance at high resolution
            print(f"Cell {idx} (Row {row+1}, Col {col+1}): {cell.shape[1]}x{cell.shape[0]} → {target_size}x{target_size}")
            
            if enhancement_mode == 'sharp':
                enhanced_cell = sharp_clean_enhancement(cell, target_size)
            else:
                enhanced_cell = ultra_clean_enhancement(cell, target_size)
            
            if enhanced_cell is not None:
                # Save enhanced cell
                filename = f"cell_{idx:03d}_highres_{enhancement_mode}_r{row+1:02d}_c{col+1:02d}.jpg"
                output_path = Path(output_folder) / filename
                cv2.imwrite(str(output_path), enhanced_cell, [cv2.IMWRITE_JPEG_QUALITY, 98])
                
                enhanced_cells.append({
                    'image': enhanced_cell,
                    'row': row,
                    'col': col,
                    'index': idx,
                    'filename': filename
                })
                print(f"    ✓ Saved: {filename}")
            
        except Exception as e:
            print(f"    ✗ Error processing cell {idx}: {e}")
    
    print(f"✅ Enhanced {len(enhanced_cells)} specific cells")
    return enhanced_cells

def batch_process_multiple_panels(input_folder, output_base_folder, target_size=512, enhancement_mode='sharp'):
    """
    Process multiple solar panel images in batch
    """
    # Find all image files
    input_path = Path(input_folder)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No image files found in {input_folder}")
        return
    
    print(f"🔄 Batch processing {len(image_files)} images with '{enhancement_mode}' mode...")
    
    for i, img_path in enumerate(image_files):
        try:
            print(f"\n📷 Processing image {i+1}/{len(image_files)}: {img_path.name}")
            
            # Create output folder for this image
            img_name = img_path.stem
            output_folder = Path(output_base_folder) / f"{img_name}_enhanced_cells_{enhancement_mode}"
            
            # Process all cells
            enhanced_cells = extract_and_enhance_all_cells(
                str(img_path), str(output_folder), target_size, enhancement_mode=enhancement_mode
            )
            
            # Create overview if we have cells
            if enhanced_cells:
                create_cells_overview(enhanced_cells, str(output_folder))
                analyze_cell_quality(enhanced_cells)
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
    
    print(f"\n🎉 Batch processing completed!")
    print(f"📁 All results saved to: {output_base_folder}")

# Main execution function
def main():
    """
    Main function with example usage
    """
    # Configuration
    INPUT_IMAGE = "/home/flo/Videos/EL project/EL_Test/SH151009P636KSPC-521.jpg"  # Change this to your image path
    OUTPUT_FOLDER = "enhanced_solar_cells_sharp"
    TARGET_SIZE = 512  # Size of each enhanced cell
    GRID_ROWS = 6      # Adjust based on your panel
    GRID_COLS = 10     # Adjust based on your panel
    ENHANCEMENT_MODE = 'sharp'  # 'sharp', 'balanced', or 'clean'
    
    print("🔬 Solar Panel Cell Enhancement System - SHARP MODE")
    print("=" * 60)
    print("Specialized for maximum sharpness and defect detection")
    print("=" * 60)
    
    try:
        # Step 1: Quick test on single cell
        print("\n🧪 STEP 1: Quick single cell test with sharp vs balanced comparison")
        test_result = quick_single_cell_test(INPUT_IMAGE, cell_index=15, target_size=1024, enhancement_mode=ENHANCEMENT_MODE)
        
        if test_result is not None:
            print("✅ Quick test successful!")
            
            # Step 2: Process all cells with SHARP enhancement
            print(f"\n🏭 STEP 2: Processing all cells with SHARP enhancement")
            enhanced_cells = extract_and_enhance_all_cells(
                INPUT_IMAGE, OUTPUT_FOLDER, TARGET_SIZE, GRID_ROWS, GRID_COLS, enhancement_mode=ENHANCEMENT_MODE
            )
            
            if enhanced_cells:
                # Step 3: Create overview and analysis
                print(f"\n📊 STEP 3: Creating overview and analysis")
                create_cells_overview(enhanced_cells, OUTPUT_FOLDER, GRID_ROWS, GRID_COLS)
                analyze_cell_quality(enhanced_cells)
                
                # Step 4: High-resolution specific cells (optional)
                print(f"\n🔍 STEP 4: High-resolution enhancement of specific cells")
                specific_cells = [5, 15, 25, 35, 45]  # Example cell indices
                high_res_cells = enhance_specific_cells(
                    INPUT_IMAGE, f"{OUTPUT_FOLDER}_high_res", specific_cells, 
                    target_size=1024, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, enhancement_mode=ENHANCEMENT_MODE
                )
                
                print(f"\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
                print(f"📁 Output folders:")
                print(f"   - Standard cells:  {OUTPUT_FOLDER}/")
                print(f"   - High-res cells:  {OUTPUT_FOLDER}_high_res/")
                print(f"   - Overview image:  {OUTPUT_FOLDER}/enhanced_cells_overview.png")
                print(f"   - Enhancement mode: {ENHANCEMENT_MODE.upper()}")
                
                print(f"\n📈 Summary:")
                print(f"   - Total cells processed: {len(enhanced_cells)}")
                print(f"   - Cell size: {TARGET_SIZE}x{TARGET_SIZE} pixels")
                print(f"   - High-res cells: {len(high_res_cells)} at 1024x1024")
                print(f"   - Quality: Maximum sharpness for AI detection")
            
        else:
            print("❌ Quick test failed. Please check your image path.")
    
    except Exception as e:
        print(f"❌ Error in main processing: {e}")
        print("Please check if the image file exists and the path is correct")

# Utility functions for different use cases
def process_single_image(image_path, output_folder="enhanced_cells_sharp", target_size=512, enhancement_mode='sharp'):
    """Simple function to process a single image with sharp enhancement"""
    enhanced_cells = extract_and_enhance_all_cells(image_path, output_folder, target_size, enhancement_mode=enhancement_mode)
    if enhanced_cells:
        create_cells_overview(enhanced_cells, output_folder)
        analyze_cell_quality(enhanced_cells)
    return enhanced_cells

def test_enhancement_quality(image_path, cell_index=10, enhancement_mode='sharp'):
    """Test enhancement quality on a single cell with sharp mode"""
    return quick_single_cell_test(image_path, cell_index, target_size=1024, enhancement_mode=enhancement_mode)

# Example usage
if __name__ == "__main__":
    # Run main function
    main()
    
    # Uncomment below for other use cases:
    
    # # Process single image quickly with sharp mode
    # process_single_image("your_image.jpg", "output_cells_sharp", 512, 'sharp')
    
    # # Test specific cell with sharp enhancement
    # test_enhancement_quality("your_image.jpg", cell_index=20, 'sharp')
    
    # # Batch process multiple images with sharp mode
    # batch_process_multiple_panels("input_images/", "output_results/", 512, 'sharp')            {sharp_sharpness:.1f} ({(sharp_sharpness/original_sharpness-1)*100:+.1f}%)")
       