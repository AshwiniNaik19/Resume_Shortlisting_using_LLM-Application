# Imports 
from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
import traceback

app = Flask(__name__)

UPLOAD_API = "http://localhost:8000/upload-resumes/"
CSV_FILE = "resume_extraction_results.csv"
REMOVED_RESUMES_PATH = r"Removed_Resumes/Removed_Resumes_Data.csv"

all_results = []
duplicate_results = []

@app.route('/')
def home():
    return redirect(url_for('leaderboard'))

# ---------- Leaderboard ----------
@app.route('/leaderboard')
def leaderboard():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = df.fillna("N/A")
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        resumes = df.to_dict(orient="records")
        sorted_resumes = sorted(resumes, key=lambda x: x.get('score', 0), reverse=True)
    else:
        sorted_resumes = []
    return render_template('leaderboard.html', resumes=sorted_resumes)

# ---------- Upload Page (GET) ----------
@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')

# ---------- Upload Resume (POST) ----------
@app.route('/upload', methods=['POST'])
def upload_resume():
    global all_results, duplicate_results
    files = request.files.getlist('resumes')
    file_data = [('files', (file.filename, file.read(), file.content_type)) for file in files]

    try:
        response = requests.post(UPLOAD_API, files=file_data)
        if response.status_code == 200:
            data = response.json()
            all_results = data.get("all_results", [])
            duplicate_results = data.get("duplicates", [])
            return redirect(url_for('show_results'))
        else:
            return f"Backend error: {response.text}"
    except Exception as e:
        return f"Request failed: {e}"

# ---------- Show Upload Results ----------
@app.route('/results')
def show_results():
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    return render_template('results.html', resumes=sorted_results, duplicates=duplicate_results)

# ---------- Remove Resume from UI ----------
@app.route('/remove-resume-ui', methods=['DELETE'])
def remove_resume_ui():
    identifier = request.args.get('identifier')
    if not identifier:
        return jsonify({'status': 'fail', 'message': 'Missing identifier'}), 400

    try:
        removal_reason = request.get_json().get("removal_reason")
        response = requests.delete(
            "http://localhost:8000/remove-resume/",
            params={"identifier": identifier},
            json={"removal_reason": removal_reason}
        )
        if response.status_code == 200:
            return jsonify({'status': 'success', 'message': response.json().get("message")})
        else:
            return jsonify({'status': 'fail', 'message': response.json().get("detail", "Unknown error")})
    except Exception as e:
        return jsonify({'status': 'fail', 'message': str(e)}), 500

# ---------- Discarded Resumes ----------
@app.route("/discarded-resumes")
def discarded_resumes():
    try:
        if not os.path.exists(REMOVED_RESUMES_PATH):
            return render_template("discarded_resumes.html", data=[], is_empty=True)

        df = pd.read_csv(REMOVED_RESUMES_PATH)
        is_empty = df.empty
        return render_template("discarded_resumes.html", data=df.to_dict(orient="records"), is_empty=is_empty)

    except Exception as e:
        return f"Error loading discarded resumes: {e}"

# ===================== Run App =====================
if __name__ == '__main__':
    app.run(debug=True, port=8001)
