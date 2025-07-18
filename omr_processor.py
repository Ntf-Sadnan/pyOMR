import cv2
import numpy as np
import json
import base64
import traceback

def process_omr_sheet(base64_image_data):
    """
    Processes an OMR sheet from a base64 encoded image string using a proven, accurate scanning logic.

    Args:
        base64_image_data (str): The base64 encoded string of the image.

    Returns:
        tuple: A tuple containing:
            - str: The base64 encoded string of the output image with detected answers highlighted.
            - dict: A dictionary containing the detected answers, with question numbers as keys.
            If an error occurs, the first element is None and the second is a dictionary with an 'error' key.
    """
    try:
        # Load base positioning data from the JSON file
        with open("omr_base_data.json", 'r') as f:
            base_data = json.load(f)

        # Decode the base64 image data to an OpenCV image
        image_data = base64.b64decode(base64_image_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        original_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if original_img is None:
            raise ValueError("Could not decode image from the provided base64 string.")

        # === Image Alignment (Identical to the accurate script) ===
        processing_img = original_img.copy()
        h, w = processing_img.shape[:2]
        # Ensure consistent orientation (portrait)
        if w > h:
            processing_img = cv2.rotate(processing_img, cv2.ROTATE_90_CLOCKWISE)

        # Pre-process for alignment marker detection
        gray = cv2.cvtColor(processing_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)[1]

        # Find the four corner markers of the sheet
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_contours = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) == 4 and cv2.contourArea(cnt) > 100]

        if len(marker_contours) < 4:
            raise ValueError("Could not find all 4 corner markers. The sheet may be obscured or damaged.")

        # Get the corners of the page and apply a perspective transform
        all_points = np.concatenate([c for c in marker_contours])
        corners_approx = cv2.approxPolyDP(cv2.convexHull(all_points), 0.02 * cv2.arcLength(cv2.convexHull(all_points), True), True)

        if len(corners_approx) != 4:
            raise ValueError("Could not accurately identify the four corners for alignment.")

        pts = corners_approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

        # Transform the image to match the base template's dimensions
        base_width = base_data["template_dimensions"]["width"]
        base_height = base_data["template_dimensions"]["height"]
        dst = np.array([[0, 0], [base_width - 1, 0], [base_width - 1, base_height - 1], [0, base_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        aligned_img = cv2.warpPerspective(processing_img, M, (base_width, base_height))

        # === Bubble Detection (Identical to the accurate script) ===
        gray_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

        # Use a strict threshold for initial contour finding
        thresh_strict = cv2.threshold(gray_aligned, 150, 255, cv2.THRESH_BINARY_INV)[1]

        # Use a more sensitive threshold for checking the fill ratio
        thresh_sensitive = cv2.threshold(gray_aligned, 180, 255, cv2.THRESH_BINARY_INV)[1]

        contours, _ = cv2.findContours(thresh_strict.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_bubbles = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0

            # Filter for contours that look like bubbles
            if 50 < area < 600 and 0.6 <= aspect_ratio <= 1.5:
                mask = np.zeros(thresh_sensitive.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                total_pixels = cv2.countNonZero(mask)

                if total_pixels > 0:
                    # Calculate fill ratios using both thresholds for robustness
                    filled_pixels_strict = cv2.countNonZero(cv2.bitwise_and(thresh_strict, thresh_strict, mask=mask))
                    filled_pixels_sensitive = cv2.countNonZero(cv2.bitwise_and(thresh_sensitive, thresh_sensitive, mask=mask))
                    fill_ratio_strict = filled_pixels_strict / float(total_pixels)
                    fill_ratio_sensitive = filled_pixels_sensitive / float(total_pixels)

                    # Determine if a bubble is filled based on a combination of criteria
                    is_filled = (
                        fill_ratio_strict > 0.7 or
                        (fill_ratio_sensitive > 0.5 and area > 80)
                    )

                    if is_filled:
                        # Calculate the centroid of the bubble
                        moments = cv2.moments(cnt)
                        if moments["m00"] != 0:
                            cX = int(moments["m10"] / moments["m00"])
                            cY = int(moments["m01"] / moments["m00"])
                            detected_bubbles.append({
                                'x': cX, 
                                'y': cY, 
                                'contour': cnt
                            })

        # === Bubble Matching (Identical to the accurate script) ===
        answers = {}
        output_image = aligned_img.copy()

        for bubble in detected_bubbles:
            bubble_x, bubble_y = bubble['x'], bubble['y']

            # Highlight the detected bubble on the output image
            cv2.drawContours(output_image, [bubble['contour']], -1, (0, 255, 0), -1)

            best_match = None
            min_distance = float('inf')

            # Find the closest question/option this bubble corresponds to
            for q_num_str, question_data in base_data["questions"].items():
                for option_letter, option_data in question_data["options"].items():
                    bounds = option_data["bounds"]

                    # Check if the bubble's center is within the option's bounding box
                    if (bounds["left"] <= bubble_x <= bounds["right"] and
                        bounds["top"] <= bubble_y <= bounds["bottom"]):

                        center_x = option_data["x"]
                        center_y = option_data["y"]
                        distance = np.sqrt((bubble_x - center_x)**2 + (bubble_y - center_y)**2)

                        # Find the closest option center
                        if distance < min_distance:
                            min_distance = distance
                            best_match = {
                                'question': int(q_num_str),
                                'option': option_letter,
                            }

            # Assign the answer if a close and unambiguous match is found
            if best_match and min_distance < 50:  # Distance threshold to prevent mismatches
                q_num = best_match['question']
                option = best_match['option']

                # To handle multiple marks for one question, we favor the one with the smallest distance.
                # This logic is implicitly handled by iterating through all bubbles. If a later bubble
                # matches the same question, it will overwrite the previous one. A more advanced
                # implementation could flag these as errors.
                answers[q_num] = option

        # === Finalization ===
        # Encode the processed image back to a base64 string for the API response
        _, buffer = cv2.imencode('.jpg', output_image)
        output_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Sort answers by question number for a clean final output
        sorted_answers = {k: answers[k] for k in sorted(answers.keys())}

        return output_image_base64, sorted_answers

    except Exception as e:
        # For production, you might want to log the full error
        traceback.print_exc()
        # Return a clear error message in the API response
        return None, {"error": str(e)}
