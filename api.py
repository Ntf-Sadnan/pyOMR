from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import subprocess
import uuid
import traceback

# Import from your existing omr_processor.py
from omr_processor import process_omr_sheet

app = Flask(__name__)
CORS(app) 

@app.route('/scan', methods=['POST'])
def scan_omr():
    """
    API endpoint to scan an OMR sheet by executing scanfirst.py
    and then passing its output to the OMR processor.
    """
    if not request.json or 'base64_data' not in request.json:
        return jsonify({"error": "Missing 'base64_data' in request body"}), 400

    base64_data = request.json['base64_data']

    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}_input.jpg"
    scanned_filename = f"{unique_id}_input_scanned.jpg" # This matches the naming from scanfirst.py

    try:
        # STAGE 1: Save the incoming image to a file
        print(f"[INFO] Saving incoming image to {input_filename}")
        image_data = base64.b64decode(base64_data)
        with open(input_filename, 'wb') as f:
            f.write(image_data)
        print(f"[INFO] Successfully saved {input_filename}")

        # STAGE 2: Execute scanfirst.py using a command-line call
        command = ["python", "scanfirst.py", "--image", input_filename]
        print(f"[INFO] Executing command: {' '.join(command)}")

        process_result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"[INFO] scanfirst.py executed successfully.")
        print(f"       STDOUT: {process_result.stdout.strip()}")

        # STAGE 3: Load the resulting _scanned.jpg file
        print(f"[INFO] Attempting to load output file: {scanned_filename}")
        if not os.path.exists(scanned_filename):
            print(f"[ERROR] Output file '{scanned_filename}' not found after script execution!")
            return jsonify({
                "error": f"scanfirst.py did not produce the expected output file: {scanned_filename}",
                "details": process_result.stdout
            }), 500

        print(f"[INFO] Found {scanned_filename}. Reading its content.")
        with open(scanned_filename, 'rb') as f:
            scanned_image_bytes = f.read()

        processed_base64_data = base64.b64encode(scanned_image_bytes).decode('utf-8')
        print(f"[INFO] Content of {scanned_filename} encoded to base64.")

        # STAGE 4: Send the new image data to the omr_processor
        print("[INFO] Passing the scanned image to omr_processor.py...")
        scanned_image_base64, omr_data = process_omr_sheet(processed_base64_data)
        print("[INFO] Received final results from omr_processor.py.")

        if "error" in omr_data:
            return jsonify(omr_data), 500

        # Final Response
        response_data = {
            "scanned_results.jpg": scanned_image_base64,
            "omr_base_data.json": omr_data
        }

        return jsonify(response_data)

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Execution of scanfirst.py failed!")
        print(f"        STDOUT: {e.stdout}")
        print(f"        STDERR: {e.stderr}")
        return jsonify({
            "error": "Execution of the scanfirst.py script failed.",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred in the API: {str(e)}"}), 500

    finally:
        # Cleanup: Delete the temporary files
        print("[INFO] Cleaning up temporary files...")
        for f in [input_filename, scanned_filename]:
            if os.path.exists(f):
                os.remove(f)
                print(f"       Removed {f}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
