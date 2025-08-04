from flask import Flask, request, jsonify, send_file
import os
import glob
from datetime import datetime
from train_cnn.classify_nbi_image import classify_nbi_image
from fpdf import FPDF
import locale
from datetime import date
from flask_cors import CORS
import json
from urllib.parse import unquote


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load patient history data
try:
    with open('patient_data/patient_data.json', 'r') as f:
        patient_data = json.load(f)
except Exception as e:
    patient_data = {}
    print("Failed to load patient history:", e)


def get_interpreted_diagnosis(scores):
    n = scores.get("Normal", 0)
    ag = scores.get("AG", 0)
    im = scores.get("IM", 0)
    dy = scores.get("Dysplasia", 0)
    ca = scores.get("Cancer", 0)

    if ca >= 45:
        return "Cancer"
    elif 25 < ca < 45:
        return "Start of Cancer"
    elif dy > 30:
        return "Dysplasia"
    elif im > 50:
        return "IM"
    elif ag + im > 80 and im > 30:
        return "AG with early signs of IM"
    elif ag > 50:
        return "AG"
    elif n + ag > 80 and ag > 30:
        return "Start of AG"
    elif n > 90:
        return "Normal"
    
    # Fallback: use dominant score with priority on ties
    score_dict = {
        "Cancer": ca,
        "Dysplasia": dy,
        "IM": im,
        "AG": ag,
        "Normal": n
    }

    # Sort by value descending, then by priority
    priority = {"Cancer": 0, "Dysplasia": 1, "IM": 2, "AG": 3, "Normal": 4}
    sorted_items = sorted(score_dict.items(), key=lambda x: (-x[1], priority[x[0]]))
    top_label, top_score = sorted_items[0]
    if top_score > 0:
        return top_label

    # Final fallback
    return "Uncertain"

@app.route('/classify', methods=['POST'])
def classify():
    uploaded = request.files['image']
    name = request.form.get('name', 'Unknown')

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + uploaded.filename
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded.save(img_path)

    # Appel du modèle
    vis_path, scores, confidence = classify_nbi_image(img_path, model_name="efficientnetb4")
    scores = {k: float(scores[k]) for k in scores}
    diagnosis = get_interpreted_diagnosis(scores)

    # Nettoyage des fichiers uploadés
    for f in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to remove {f}: {e}")

    # Clean old PDF reports
    for f in glob.glob(os.path.join(RESULT_FOLDER, '*.pdf')):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to remove {f}: {e}")

    return jsonify({
        'scores': scores,
        'diagnosis': diagnosis,
        'result_img': '/result.png',
        'confidence': round(confidence, 2),
        'name': name
    })


@app.route('/result.png')
def result_img():
    return send_file('results/result.png', mimetype='image/png')


@app.route('/generate_report', methods=['GET'])
def generate_report():
    name = unquote(request.args.get('name', 'Unknown'))
    diagnosis = request.args.get('diagnosis', 'Uncertain')
    print("Received name from frontend:", repr(name))

    explanations = {
        'Normal': (
            "The examined gastric mucosa appears normal. "
            "There are no detectable big signs of inflammation, atrophy, metaplasia, dysplasia, or gastric cancer. "
            "No immediate clinical concern is indicated based on the analyzed image."
        ),

        'Start of AG': (
            "The mucosa appears largely normal, but early signs consistent with atrophic changes are emerging. "
            "These may indicate the initial stage of Atrophic Gastritis (AG), a chronic condition involving the thinning of the stomach lining. "
            "Routine monitoring is advised to track potential progression."
        ),

        'AG': (
            "Atrophic Gastritis (AG) has been detected, characterized by the chronic inflammation and thinning of the gastric mucosa. "
            "AG can lead to a loss of glandular cells and may serve as a precursor to intestinal metaplasia. "
            "Endoscopic surveillance and clinical follow-up are recommended."
        ),

        'AG with early signs of IM': (
            "The analysis indicates the presence of Atrophic Gastritis (AG) along with early signs of Intestinal Metaplasia (IM), "
            "a condition where the stomach lining begins to resemble intestinal tissue. "
            "This progression increases the risk of neoplastic transformation. Closer clinical monitoring is advised."
        ),

        'IM': (
            "Intestinal Metaplasia (IM) is detected, indicating a transformation of gastric epithelial cells into intestinal-type cells. "
            "IM is considered a premalignant condition and may increase the long-term risk of gastric cancer. "
            "Endoscopic and histological follow-up is strongly recommended."
        ),

        'Dysplasia': (
            "Gastric dysplasia is detected, which refers to the abnormal development of epithelial cells. "
            "This is considered a high-risk precancerous lesion that may progress to adenocarcinoma. "
            "Immediate follow-up with targeted biopsies and histopathological confirmation is required."
        ),

        'Start of Cancer': (
            "Early features suggestive of gastric cancer have been identified. "
            "While the lesion may still be in its initial stage, malignant transformation is likely underway. "
            "Prompt diagnostic confirmation and treatment planning are highly recommended."
        ),

        'Cancer': (
            "The analysis reveals definitive signs of gastric cancer. "
            "This includes abnormal architecture and cellular features consistent with malignancy. "
            "Urgent medical evaluation and oncological intervention are necessary without delay."
        ),

        'Uncertain': (
            "The AI system could not confidently assign a diagnostic category based on the image. "
            "This may be due to image quality limitations or overlapping visual features. "
            "A second analysis or clinical review is advised."
        )
    }

    message = explanations.get(diagnosis, "No specific explanation found.")

    history_lines = []
    print("Searching in patient_data names:")
    for patient in patient_data:
        print("  -", repr(patient.get('name', '')))

    # Try to find matching patient in the list
    for patient in patient_data:
        if patient.get('name', '').strip().lower() == name.strip().lower():
            print("MATCH FOUND:", patient['name'])
            for record in patient.get('records', []):
                line = f"- {record['date']}: {record['diagnosis']} → {record['comments']}"
                history_lines.append(line)
            break



    pdf = FPDF(format='A4')
    pdf.add_page()

    # Background image (must exist)
    bg_path = os.path.join('frontend/src/assets/', 'reportbg.png')
    if os.path.exists(bg_path):
        pdf.image(bg_path, x=0, y=0, w=210, h=297)

    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(200, 10, txt="EndoAI - Medical Report", ln=1, align='C')
    pdf.ln(20)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient: {name}", ln=1)
    pdf.cell(200, 10, txt=f"Main diagnosis: {diagnosis}", ln=1)

    try:
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    except locale.Error:
        locale.setlocale(locale.LC_TIME, 'English')
    pdf.cell(200, 10, txt=f"Date: {date.today().strftime('%B %d, %Y')}", ln=1)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Remarks:\n{message}")
    pdf.ln(8)

    if history_lines:
        pdf.multi_cell(0, 8, "History of patient:\n" + "\n".join(history_lines))
    else:
        pdf.multi_cell(0, 10, "History of patient: unknown")


    report_path = os.path.join(RESULT_FOLDER, f"{name}_report.pdf")
    pdf.output(report_path)
    return send_file(report_path, as_attachment=True)


    
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    organization = data.get('organization')
    username = data.get('username')
    password = data.get('password')

    with open('users.json', 'r') as f:
        users = json.load(f)["users"]  # <<== FIXED

    for user in users:
        if (user['organization'] == organization and
            user['username'] == username and
            user['password'] == password):
            return jsonify({"success": True})

    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/organizations', methods=['GET'])
def get_organizations():
    try:
        with open('users.json', 'r') as f:
            data = json.load(f)
        users = data.get("users", [])
        organizations = sorted(set(user.get("organization") for user in users if "organization" in user))
        return jsonify(organizations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
