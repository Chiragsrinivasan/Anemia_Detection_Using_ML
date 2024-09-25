import os
from eye_anemia_detection import anemia_detection_eye
from palm_anemia_detection import anemia_detection_palm

def detect_anemia(input_folder):
    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print("Error: Input folder not found.")
        return

    # Check if there are any image files in the input folder
    image_files = [file for file in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file)) and file.lower().startswith(('eye', 'palm')) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print("Error: No image files found in the input folder.")
        return

    # Print the list of image files found
    print("Image files found in the input folder:")
    for file in image_files:
        print(file)

    # Check if both Eye and Palm images are present
    if not any(file.lower().startswith('eye') for file in image_files) or not any(file.lower().startswith('palm') for file in image_files):
        print("Error: Both Eye and Palm images not found in the input folder.")
        return

    # Perform anemia detection for eye image
    eye_image_path = os.path.join(input_folder, [file for file in image_files if file.lower().startswith('eye')][0])
    eye_result = anemia_detection_eye(eye_image_path)
    eye_anemia = 1 if "Anemia Detected" in eye_result else 0  # Convert result to 1 (anemic) or 0 (non-anemic)
    print("Eye Anemia Detection Result:", eye_result)

    # Perform anemia detection for palm image
    palm_image_path = os.path.join(input_folder, [file for file in image_files if file.lower().startswith('palm')][0])
    palm_result = anemia_detection_palm(palm_image_path)
    palm_anemia = 1 if "Anemic" in palm_result else 0  # Convert result to 1 (anemic) or 0 (non-anemic)
    print("Palm Anemia Detection Result:", palm_result)

    # Summarize and show results
    print("\nSummary:")
    print(f"Eye Anemia: {eye_anemia}")  # Show 0 or 1 based on eye result
    print(f"Palm Anemia: {palm_anemia}")  # Show 0 or 1 based on palm result
    Result = .7 * eye_anemia + .3 * palm_anemia


if __name__ == "__main__":
    input_folder = input("Enter the path to the folder containing Eye and Palm images: ")
    detect_anemia(input_folder)
