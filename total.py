from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from anemia_detection_controller import detect_anemia_in_folder
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_TEMPLATES'] = {
    'Anemia Detected': 'anemic.html',
    'No Anemia Detected': 'non_anemic.html'
}

# Map symptom names to their corresponding form field names
SYMPTOM_FIELD_MAP = {
    'fatigue': 'symptoms1',
    'weakness': 'symptoms2',
    'shortness_breath': 'symptoms3',
    'dizziness': 'symptoms4',
    'pale_skin': 'symptoms5'
}

def get_selected_symptoms(request_form):
    selected_symptoms = []
    for symptom, field_name in SYMPTOM_FIELD_MAP.items():
        if field_name in request_form:
            selected_symptoms.append(symptom)
    return selected_symptoms

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    age = int(request.form['age']) if request.form.get('age') else None
    gender = request.form['gender'] if request.form.get('gender') else None

    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    os.makedirs(folder_path, exist_ok=True)

    uploaded_files = []
    for idx, file in enumerate(request.files.getlist('file')):
        if file.filename != '':
            filename = secure_filename(file.filename)
            # Define the file name based on the index
            if idx == 0:
                file_name = 'Eye.jpg'
            elif idx == 1:
                file_name = 'Palm.jpg'
            elif idx == 2:
                file_name = 'Nail.jpg'
            else:
                file_name = filename
            file_path = os.path.join(folder_path, file_name)
            file.save(file_path)
            uploaded_files.append(file_path)

    # Extract selected symptoms
    selected_symptoms = get_selected_symptoms(request.form)

    # Call detect_anemia_in_folder function with the image folder path and additional information
    overall_result = detect_anemia_in_folder(folder_path, age, gender, selected_symptoms)

    # Save results to JSON file
    data = {
        'name': name,
        'overall_result': overall_result
    }
    json_filename = os.path.join(folder_path, name + '.json')
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file)

    # Determine the result and redirect to the appropriate page
    if overall_result is not None:
        if overall_result == 'Anemia Detected':
            return redirect(url_for('result', result='Anemia Detected'))
        else:
            return redirect(url_for('result', result='No Anemia Detected'))
    else:
        return "Error occurred during anemia detection."

@app.route('/result/<result>')
def result(result):
    template = app.config['RESULT_TEMPLATES'].get(result, 'non_anemic.html')
    return render_template(template)

if __name__ == '__main__':
    app.run(debug=True)
