import cv2
import numpy as np
import sys

def find_registration_markers(image):
    """Find the 6 black square registration markers"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Binary threshold to find black squares
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for square-like shapes
    squares = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Look for medium-sized squares (registration markers)
        if 200 < area < 5000:
            # Check if it's roughly square
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.3:  # Roughly square
                squares.append((x + w//2, y + h//2, area))  # Center point and area

    # Sort by area and take the 6 largest squares
    squares = sorted(squares, key=lambda x: x[2], reverse=True)[:6]

    if len(squares) < 4:
        return None

    # Get just the center points
    points = [(x, y) for x, y, _ in squares]
    return np.array(points, dtype=np.float32)

def find_bounding_rectangle(points):
    """Find the bounding rectangle that contains all registration markers"""
    if points is None or len(points) < 4:
        return None

    # Find min/max coordinates with some padding
    min_x = int(np.min(points[:, 0])) - 50
    max_x = int(np.max(points[:, 0])) + 50
    min_y = int(np.min(points[:, 1])) - 50
    max_y = int(np.max(points[:, 1])) + 50

    return (min_x, min_y, max_x, max_y)

def crop_to_markers(image, bbox):
    """Crop image to the bounding box containing all markers"""
    if bbox is None:
        return image

    min_x, min_y, max_x, max_y = bbox
    h, w = image.shape[:2]

    # Ensure boundaries are within image
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(w, max_x)
    max_y = min(h, max_y)

    return image[min_y:max_y, min_x:max_x]

def enhance_contrast_preserving_quality(image):
    """High-quality contrast enhancement without quality loss"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image.copy()

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    enhanced_l = clahe.apply(l)

    if len(image.shape) == 3:
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        return enhanced_l

def high_quality_scan(image):
    """Professional quality document scanning without quality loss"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    processed = enhance_contrast_preserving_quality(image)
    processed = cv2.bilateralFilter(processed, 9, 75, 75)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(processed, -1, kernel)
    processed = cv2.addWeighted(processed, 0.7, sharpened, 0.3, 0)

    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    return processed

def otsu_with_preprocessing(image):
    """Otsu thresholding with preprocessing for clean binary output"""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--image":
        print("Usage: python main.py --image filename.jpg")
        sys.exit(1)

    image_path = sys.argv[2]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)

    print("Detecting registration markers...")

    # Find the 6 black square registration markers
    markers = find_registration_markers(image)

    if markers is not None:
        print(f"Found {len(markers)} registration markers")

        # Get bounding box for all markers
        bbox = find_bounding_rectangle(markers)

        # Crop to the area containing all markers
        cropped = crop_to_markers(image, bbox)
        print(f"Cropped from {image.shape[1]}x{image.shape[0]} to {cropped.shape[1]}x{cropped.shape[0]}")

        # Process the cropped image
        enhanced = high_quality_scan(cropped)
        result = otsu_with_preprocessing(enhanced)
    else:
        print("Could not find registration markers, processing full image...")
        enhanced = high_quality_scan(image)
        result = otsu_with_preprocessing(enhanced)

    # Generate output filename
    base_name = image_path.rsplit('.', 1)[0]
    extension = image_path.rsplit('.', 1)[1] if '.' in image_path else 'jpg'
    output_path = f"{base_name}_scanned.{extension}"

    # Save with high quality
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Scanned document saved to {output_path}")
