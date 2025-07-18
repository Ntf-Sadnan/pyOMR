import cv2
import numpy as np
import json
import base64

def process_omr_sheet(base64_image_data):
    """
    Processes an OMR sheet from a base64 encoded image string.

    Args:
        base64_image_data (str): The base64 encoded string of the image.

    Returns:
        tuple: A tuple containing:
            - str: The base64 encoded string of the output image with scanned results.
            - dict: A dictionary containing the detected answers.
    """
    try:
        # Load base positioning data
        with open("omr_base_data.json", 'r') as f:
            base_data = json.load(f)

        # Decode the base64 image
        image_data = base64.b64decode(base64_image_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        original_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if original_img is None:
            raise ValueError("Could not decode image from base64 string.")

        # Apply same alignment as base generator
        processing_img = original_img.copy()
        h, w = processing_img.shape[:2]
        if w > h:
            processing_img = cv2.rotate(processing_img, cv2.ROTATE_90_CLOCKWISE)

        gray = cv2.cvtColor(processing_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)[1]

        # Find corner markers
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_contours = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) == 4 and cv2.contourArea(cnt) > 100]

        if len(marker_contours) < 4:
            raise ValueError("Could not find corner markers in user image.")

        # Get corners and apply perspective transform
        all_points = np.concatenate([c for c in marker_contours])
        corners_approx = cv2.approxPolyDP(cv2.convexHull(all_points), 0.02 * cv2.arcLength(cv2.convexHull(all_points), True), True)

        if len(corners_approx) != 4:
            raise ValueError("Could not identify corners for alignment.")

        pts = corners_approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

        # Transform to match base dimensions
        base_width = base_data["template_dimensions"]["width"]
        base_height = base_data["template_dimensions"]["height"]

        dst = np.array([[0, 0], [base_width - 1, 0], [base_width - 1, base_height - 1], [0, base_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        aligned_img = cv2.warpPerspective(processing_img, M, (base_width, base_height))

        # Detect bubbles in aligned image
        gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
        bubble_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find all potential bubbles
        detected_bubbles = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0

            if 50 < area < 600 and 0.6 <= aspect_ratio <= 1.5:
                mask = np.zeros(bubble_thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                total_pixels = cv2.countNonZero(mask)

                if total_pixels > 0:
                    filled_pixels_strict = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                    filled_pixels_sensitive = cv2.countNonZero(cv2.bitwise_and(bubble_thresh, bubble_thresh, mask=mask))

                    fill_ratio_strict = filled_pixels_strict / float(total_pixels)
                    fill_ratio_sensitive = filled_pixels_sensitive / float(total_pixels)

                    is_filled = (
                        fill_ratio_strict > 0.7 or
                        (fill_ratio_sensitive > 0.5 and area > 80)
                    )

                    if is_filled:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            detected_bubbles.append({
                                'x': cX,
                                'y': cY,
                                'area': area,
                                'fill_ratio': fill_ratio_sensitive,
                                'contour': cnt
                            })

        # Match bubbles to base positions
        answers = {}
        output_image = aligned_img.copy()

        for bubble in detected_bubbles:
            bubble_x, bubble_y = bubble['x'], bubble['y']

            cv2.drawContours(output_image, [bubble['contour']], -1, (0, 255, 0), -1)

            best_match = None
            min_distance = float('inf')

            for q_num_str, question_data in base_data["questions"].items():
                q_num = int(q_num_str)

                for option_letter, option_data in question_data["options"].items():
                    bounds = option_data["bounds"]

                    if (bounds["left"] <= bubble_x <= bounds["right"] and
                        bounds["top"] <= bubble_y <= bounds["bottom"]):

                        center_x = option_data["x"]
                        center_y = option_data["y"]
                        distance = np.sqrt((bubble_x - center_x)**2 + (bubble_y - center_y)**2)

                        if distance < min_distance:
                            min_distance = distance
                            best_match = {
                                'question': q_num,
                                'option': option_letter,
                            }

            if best_match and min_distance < 50:
                q_num = best_match['question']
                option = best_match['option']
                answers[q_num] = option

        # Encode the output image to base64
        _, buffer = cv2.imencode('.jpg', output_image)
        output_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return output_image_base64, answers

    except Exception as e:
        # In case of an error, return an error message
        return None, {"error": str(e)}
