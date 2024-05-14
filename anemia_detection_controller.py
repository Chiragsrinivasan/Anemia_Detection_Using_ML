import os
from anemia_detection import detect_anemia


def detect_anemia_in_folder(input_folder, age, gender, selected_symptoms):
    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print("Error: Input folder not found.")
        return None

    # Check if there are any image files in the input folder
    image_files = [file for file in os.listdir(input_folder) if
                   os.path.isfile(os.path.join(input_folder, file)) and file.lower().endswith(
                       ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if len(image_files) < 3:
        print("Error: At least 3 image files required in the input folder.")
        return None

    # Print the list of image files for debugging
    print("Image Files:", image_files)

    # Define weights for each part
    weights = {'eye': 0.20, 'palm': 0.20, 'nail': 0.15, 'symptoms': 0.4, 'age': 0.1,
               'gender': 0.05}  # Adjusted weights for each part

    # Accumulate results for each iteration
    accumulated_results = []

    # Run detection 10 times
    for _ in range(10):
        # Perform anemia detection for each image type
        results = {}
        for prefix in ['eye', 'palm', 'nail']:
            image_files_with_prefix = [file for file in image_files if file.lower().startswith(prefix)]
            if not image_files_with_prefix:
                print(f"No files found with prefix '{prefix}'. Skipping...")
                continue

            image_path = os.path.join(input_folder, image_files_with_prefix[0])
            results[prefix] = detect_anemia(image_path)

        # Count the number of symptoms selected
        num_symptoms_selected = len(selected_symptoms)

        # Print the filenames considered for matching 'symptoms' prefix
        symptoms_files = [file for file in image_files if file.lower().startswith('symptoms')]
        print("Symptoms Files:", symptoms_files)

        # Calculate the weighted result for symptoms based on the number of symptoms selected
        if num_symptoms_selected == 0:
            symptoms_weightage = 0 * .4  # Half of the allocated weightage
        elif num_symptoms_selected == 1:
            symptoms_weightage = .5 * .4  # Half of the allocated weightage
        elif num_symptoms_selected == 2:
            symptoms_weightage = .75 * .4  # 3/4 of the allocated weightage
        else:
            symptoms_weightage = .4  # Full weightage if 3 or more symptoms are selected

        # Calculate the individual contributions of each component
        breakdown = {}
        for prefix, result in results.items():
            breakdown[prefix] = weights[prefix] * ("Anemia Detected" in result)
        breakdown['symptoms'] = symptoms_weightage

        # Adjust the weighted result based on age and gender
        if age < 18:
            weighted_result = sum(breakdown.values()) * 1.1  # Increase weightage for individuals under 18
        else:
            weighted_result = sum(breakdown.values())
        if gender == 'Female':
            weighted_result *= 1.2  # Increase weightage for females

        # Append the weighted result to the list of accumulated results
        accumulated_results.append(weighted_result)

        # Print the breakdown for each iteration
        print("Iteration:", _ + 1)
        print("Breakdown:", breakdown)
        print("Weighted Result:", weighted_result)
        print()

    # Calculate the average of accumulated results
    average_result = sum(accumulated_results) / len(accumulated_results)

    # Print the averaged result
    print("Averaged Result:", average_result)

    # Return the final result
    return "Anemia Detected" if average_result >= .5 else "No Anemia Detected"

