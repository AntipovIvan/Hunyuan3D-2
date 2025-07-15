import requests
import time
import base64
import os
import argparse
import glob

# --- Configuration ---
# URL for your locally running FastAPI server
API_URL = "http://127.0.0.1:8080/generate"
OUTPUT_DIR = "mesh_outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Function ---
def image_to_base64_string(path):
    """Encodes an image file to a base64 string."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_set(front_path, back_path, left_path, output_filename):
    """Processes a single set of images to generate a mesh."""
    print(f"\n--- Processing set for {output_filename} ---")
    print(f"Front: {front_path}, Back: {back_path}, Left: {left_path}")

    image_paths = {
        "front_image": front_path,
        "back_image": back_path,
        "left_image": left_path,
    }

    # --- Build the Payload ---
    payload = {
        **{key: image_to_base64_string(path) for key, path in image_paths.items()},
        "seed": int(time.time()), # Use a different seed for each generation
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 10.0,
        "type": "obj",
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # --- Send Request ---
    print(f"Sending request for {output_filename}...")
    start_time = time.time()

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        end_time = time.time()
        print(f"Successfully received response in {end_time - start_time:.4f} seconds.")

        # --- Process Response ---
        response_data = response.json()
        mesh_data = base64.b64decode(response_data["mesh_base64"])

        # Save the resulting .obj file
        full_output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(full_output_path, "wb") as f:
            f.write(mesh_data)
        print(f"Saved generated mesh to '{full_output_path}'")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request for {output_filename}: {e}")
        try:
            print("Server response:", e.response.json())
        except:
            pass

# --- Main Loop ---
def main():
    # Find all front images to determine the number of sets
    front_images = sorted(glob.glob("front/front (*).png"))
    if not front_images:
        print("No front images found in 'front/' directory. Exiting.")
        print("Please ensure files are named like 'front (1).png', 'front (2).png', etc.")
        return

    print(f"Found {len(front_images)} image sets to process.")

    for i, front_image_path in enumerate(front_images, 1):
        # Extract the number from the filename, e.g., 'front (1).png' -> '1'
        try:
            num = front_image_path.split('(')[1].split(')')[0]
        except IndexError:
            print(f"Could not parse number from filename: {front_image_path}. Skipping.")
            continue

        # Construct paths for the other images in the set
        back_image_path = f"back/back ({num}).png"
        left_image_path = f"left/left ({num}).png"

        # Check if all files for the current set exist
        if os.path.exists(back_image_path) and os.path.exists(left_image_path):
            output_filename = f"{i}.obj"
            process_image_set(front_image_path, back_image_path, left_image_path, output_filename)
        else:
            print(f"\n--- Skipping set {i} ---")
            print(f"Could not find a complete image set for number {num}.")
            if not os.path.exists(back_image_path):
                print(f"Missing: {back_image_path}")
            if not os.path.exists(left_image_path):
                print(f"Missing: {left_image_path}")

if __name__ == "__main__":
    # Argument parsing is not strictly needed for this new logic, but kept for compatibility
    parser = argparse.ArgumentParser(description="Test script for Hunyuan3D API.")
    parser.add_argument('--with-texture', action='store_true', help="Generate a mesh with textures (not used in this script).")
    args = parser.parse_args()

    main()
    print("\nAll image sets processed.")
