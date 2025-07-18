from flask import Flask, request, jsonify
from omr_processor import process_omr_sheet

app = Flask(__name__)

@app.route('/scan', methods=['POST'])
def scan_omr():
    """
    API endpoint to scan an OMR sheet.
    Expects a JSON payload with a 'base64_data' key.
    """
    if not request.json or 'base64_data' not in request.json:
        return jsonify({"error": "Missing 'base64_data' in request body"}), 400

    base64_data = request.json['base64_data']
    scanned_image_base64, omr_data = process_omr_sheet(base64_data)

    if "error" in omr_data:
        return jsonify(omr_data), 500

    response_data = {
        "scanned_results.jpg": scanned_image_base64,
        "omr_base_data.json": omr_data
    }

    return jsonify(response_data)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
