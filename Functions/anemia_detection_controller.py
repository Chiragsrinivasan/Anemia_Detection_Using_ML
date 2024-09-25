import os
from anemia_detection import detect_anemia

def detect_anemia_in_folder(input_folder):
    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print("Error: Input folder not found.")
        return

    # Check if there are any image files in the input folder
    image_files = [file for file in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if len(image_files) < 3:
        print("Error: At least 3 image files required in the input folder.")
        return

    # Print the list of image files found
    print("Image files found in the input folder:")
    for file in image_files:
        print(file)

    # Perform anemia detection for each image type
    results = {}
    for prefix in ['eye', 'palm', 'nail']:
        image_path = os.path.join(input_folder, [file for file in image_files if file.lower().startswith(prefix)][0])
        results[prefix] = detect_anemia(image_path)
        print(f"{prefix.capitalize()} Anemia Detection Result:", results[prefix])

    # Define weights for each part
    weights = {'eye': 0.3, 'palm': 0.35, 'nail': 0.35}

    # Calculate the overall anemia probability
    overall_result = sum("Anemia Detected" in result for result in results.values())

    # Calculate the weighted average
    weighted_result = sum(weights[prefix] * ("Anemia Detected" in result) for prefix, result in results.items())

    # Print the overall anemia probability
    print("\nSummary:")
    print(f"Overall Anemia Probability: {weighted_result:.4f}")

if __name__ == "__main__":
    input_folder = input("Enter the path to the folder containing Eye, Palm, and Nail images: ")
    detect_anemia_in_folder(input_folder)
