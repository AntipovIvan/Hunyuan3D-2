import requests
import time
import base64
import os

# --- Configuration ---
# URL for your locally running FastAPI server
# The endpoint is "/generate" as defined in your script
API_URL = "http://127.0.0.1:8080/generate"

# --- Prepare Data ---
# This function encodes a local image file into a base64 string
def image_to_base64_string(path):
    """Encodes an image file to a base64 string."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# The keys ('front_image', 'left_image', etc.) MUST match the keys
# expected by modified api_server.py.
# Use as many views as have, up to the 6 defined in the server.
# For a real test, use 4-8 views from setup.
image_paths = {
    "front_image": "front.jpg",
    "back_image": "back.jpg",
    "left_image": "left.jpg"
}

# Check if image files exist before proceeding
for key, path in image_paths.items():
    if not os.path.exists(path):
        print(f"Error: Image file not found at '{path}'")
        print("Please update the image_paths dictionary with correct file paths.")
        exit()

# --- Build the Payload ---
# This dictionary will be sent as JSON to API.
# It includes the base64 encoded images and other generation parameters.
# This dictionary will be sent as JSON to API.
# It includes the base64 encoded images and other generation parameters.
payload = {
    # Add the base64 encoded images to the payload
    **{key: image_to_base64_string(path) for key, path in image_paths.items()},

    # --- Generation Parameters ---
    # These are passed directly to the Hunyuan3D pipeline
    "seed": 1234,
    "octree_resolution": 128,  # Higher value might improve quality but increase time
    "num_inference_steps": 5,  # Fewer steps for faster inference
    "guidance_scale": 5.0,
    "type": "obj",             # We want an .obj file as output
}

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"  # We expect a JSON response back
}

# --- Benchmarking ---
print(f"Sending multi-view request to local API at {API_URL}...")
start_time = time.time()

try:
    # Send the POST request with the JSON payload
    response = requests.post(API_URL, json=payload, headers=headers)

    # Raise an exception for bad status codes (e.g., 4xx client errors, 5xx server errors)
    response.raise_for_status()

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Successfully received response in {processing_time:.4f} seconds.")

    # --- Process the Response ---
    response_data = response.json()
    
    # Decode the mesh from base64
    mesh_data = base64.b64decode(response_data["mesh_base64"])
    
    # Save the resulting .obj file
    output_filename = response_data.get("filename", "local_api_output.obj")
    with open(output_filename, "wb") as f:
        f.write(mesh_data)
    print(f"Saved generated mesh to '{output_filename}'")

    # Print the origin
    origin = response_data.get("origin")
    if origin:
        print(f"Calculated mesh origin: {origin}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the request: {e}")
    # If the server returned a JSON error message, try to print it
    try:
        print("Server response:", e.response.json())
    except:
        pass
