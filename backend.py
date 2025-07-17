from http.client import HTTPException
import os
import re
import csv
import json
import shutil
import hashlib
import openai
from urllib.parse import urlencode
from fastapi import FastAPI, UploadFile, File, Request, FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PyPDF2 import PdfReader
import math
from ranking1 import get_company_ranks, get_degree_ranks
from ranking2 import get_college_ranks
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import date
from collections import defaultdict

# Load .env variables
load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploaded_resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploaded_resumes", StaticFiles(directory=UPLOAD_FOLDER), name="uploaded_resumes")

CSV_OUTPUT_PATH = "resume_extraction_results.csv"

def extract_text_from_pdf(file_path):
    full_text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                full_text += page.extract_text() or ""
    except Exception:
        pass
    return full_text

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"[•⋄●■◆▪▶►✓✔➤➔➥➦➧➨]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\-\+\#\(\)/@]", "", text)
    text = re.sub(r"\s([.,!?;:])", r"\1", text)
    return text.strip()

def compute_file_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def load_existing_hashes(csv_path):
    hashes = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "file_hash" in row:
                    hashes.add(row["file_hash"])
    return hashes

def load_all_records(csv_path):
    records = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    {
                        "name": row.get("name", ""),
                        "email": row.get("email", ""),
                        "processed_date": row.get("processed_date",""),
                        "phone_number": row.get("phone_number", ""),
                        "score": float(row.get("score", 0)),
                        "resume_link": row.get("resume_link", ""),
                        "skills": row.get("skills", ""),
                        "explainability": row.get("explainability", "")
                    }
                )
    return records

def validate_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$", email))

def validate_name(name):
    # Strip extra spaces and normalize spacing
    name = " ".join(name.strip().split())
    # If name is already properly capitalized, return it
    if name == name.title():
        return name
    # Else, capitalize first letter of each word
    return name.title()

def validate_list_of_strings(input_list):
    return isinstance(input_list, list) and all(
        isinstance(item, str) and item.strip() for item in input_list
    )

def invoke_gpt4(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who extracts structured resume information.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0,
            model=deployment,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("OpenAI API error:", e)
        return ""

@app.post("/upload-resumes/")
async def upload_resumes(request: Request, files: list[UploadFile] = File(...)):
    new_results = []
    duplicates = []
    existing_hashes = load_existing_hashes(CSV_OUTPUT_PATH)
    processed_hashes = set() 
    base_url = str(request.base_url)

    for file in files:
        try:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            file_hash = compute_file_hash(file_path)

            if file_hash in existing_hashes or file_hash in processed_hashes:
                duplicates.append(
                    {
                        "filename": filename,
                        "reason": "Duplicate resume (already processed)"
                    }
                )
                # os.remove(file_path)
                continue

            processed_hashes.add(file_hash)

            text = extract_text_from_pdf(file_path)
            text = clean_text(text)
            if not text:
                continue

            prompt = f"""
            Extract the following structured information from the resume text below. 
            Only extract **what is clearly present** — do not infer, generate, or guess any missing values.

            Specifically:
            - "name": extract the candidate's name 
            - "email": return the email address only if it is present; if not found, return exactly "Not Available".
            - "phone_number": return the phone_number only if it is present; if not found, return exactly "Not Available".
            - "college": extract the names of colleges attended **for Bachelor's degree or above**, but ensure the order is: 
                - Bachelor's college first 
                - then Master's or higher colleges (if present).
                If no such colleges are found, return exactly ["Not Available"].
            - "degree": extract only **Bachelor's degree with specialized in,or above**, in this order:
                - Bachelor's degrees first
                - then Master's or higher degrees (if present).
                Do not include diplomas or lower-level qualifications.
                If no such degrees are found, return exactly ["Not Available"].
            - "skills": extract skills. 
            - "company": extract only company names if clearly mentioned. If none found, return ["Not Available"].
            - "experience": if total experience is stated (e.g., "3 years 6 months"), calculate and return it as a float in **years** (e.g., 3.6). If not found, return "Not Available".
            Return the result strictly as a JSON object in the following format:
            {{
                "name": "...",
                "email": "...",
                "phone_number":"...",
                "college": [...],
                "degree":[...],
                "skills":[...],
                "company": [...],
                "experience": "..."
            }}
            Resume Text:
            \"\"\"{text}\"\"\""""

            raw_output = invoke_gpt4(prompt)
            print("LLM Raw Output:", raw_output)

            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))

                name = data.get("name", filename)
                email = data.get("email", "")
                phone_number = data.get("phone_number", "")
                college = data.get("college", [])
                degree = data.get("degree", [])
                company = data.get("company", [])
                skills = data.get("skills", [])
                experience = data.get("experience", "")
                processed_date = date.today().isoformat()
                print(name)
                print(college)
                print(degree)
                print(skills)
                print(company)
                print(processed_date)

#################### Skill mapping #########################################################
                def classify(skills):
                    prompt = f"""
                    You are an intelligent assistant that classifies resumes into the most appropriate skill category based on a list of skills.

                    There are four possible categories:

                    1. **AI_Development**: Skills include:
                    Data Structures and Algorithms, Python, REST Framework, Flask, Django, RDBMS, PostGreSQL, SQL, MYSQL, VectorDB, Prompt Engineering, LLMs (Large Language Models),LangChain,RAG (Retrieval-Augmented Generation),Embeddings,Ollama,AutoGen,AI Foundry,CrewAI,Fine-tuning LLMs,LLMOps 

                    2. **Engineering_and_Datascience**: Skills include:
                    Model Evaluation, Model Development, Distributed Computing, Python, NumPy, pandas, scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM, NLTK, SpaCy, StatsModels, Cross-validation, ROC/AUC, Precision/Recall, F1-score, Confusion Matrix, Supervised Learning, Unsupervised Learning, Machine Learning, Deep Learning, Computer Vision, OpenCV, NLP, Time Series, Reinforcement Learning, Apache Spark, Pyspark, Hadoop, Apache Kafka, Apache Airflow

                    3. **Infrastructure_and_Platform**: Skills include:
                    Infrastructure as Code (IaC), CI-CD Tools, Monitoring & Logging, Version Control, Docker, Scripting, ML Pipelines, Terraform, AWS CloudFormation, Ansible, Jenkins, GitLab CI-CD, CircleCI, GitHub Actions, Prometheus, Grafana, ELK Stack (Elasticsearch, Logstash, Kibana), Fluentd, Git, GitHub/GitLab/Bitbucket, Bash/Python/PowerShell, Mlflow, Kubeflow Pipelines, TFX, Metaflow

                    4. **Business_and_Management**: Skills include:
                    Power BI, JIRA, Powerpoints, Excel, Verbal / Written Communication

                    ---

                    Given the following extracted skills from a resume:
                    {', '.join(skills)}

                    Analyze the skills carefully. Then:
                    - Choose the **single best-matching category**.
                    - Consider the context and group of skills, not just keywords like “Python”.
                    - If none of the categories are relevant to the skills, respond exactly with:
                    No relevant skills matched the available categories: Engineering and Data Science, AI Development, Infrastructure and Platform, or Business and Management.

                    Respond with **only** one of these (no explanation):
                    - AI_Development
                    - Engineering_and_Datascience
                    - Infrastructure_and_Platform
                    - Business_and_Management
                    - No relevant skills matched the available categories: Engineering and Data Science, AI Development, Infrastructure and Platform, or Business and Management.
                    """
                    raw_output = invoke_gpt4(prompt)
                    output = f"Best Matched Skew: {raw_output.strip()}"
                    return output
                result = classify(skills)
                print("XXX",result)
#############################################################################################

                if not validate_list_of_strings(college) or not validate_list_of_strings(degree) or not validate_list_of_strings(company):
                    continue

                if not validate_email(email):
                    email = "Not Available"

                if not name or not name.strip():
                    name = "Not Available"
                else:
                    name = validate_name(name)

                college_rank = get_college_ranks(college)[0]
                ranking = get_college_ranks(college)[1]
                total = get_college_ranks(college) 
                degree_rank = get_degree_ranks(degree)
                company_rank = max(get_company_ranks(company))

                print("Before")
                print("college_rank", college_rank)
                print("degree_rank", degree_rank)
                print("company_rank", company_rank)
                print("experience", experience)
                print("ranking", ranking)

                if len(college_rank) > 0:
                    college1_rank = college_rank[0]

                if len(college_rank) > 1:
                    college2_rank = college_rank[1]
                else:
                    college2_rank = 0

                if len(degree_rank) > 0:
                    degree1_rank = degree_rank[0]

                if len(degree_rank) > 1:
                    degree2_rank = degree_rank[1]
                else:
                    degree2_rank = 0

                notice_period = 30
                exp_ctc_1 = 300000
                exp_ctc = 300000 / 100000

                total_nirf = 200
                total_qs = 1498
                experience = float(experience)
                print("After")
                print("name", name)
                print("college_rank1", college1_rank)
                print("college_rank2", college2_rank)
                print("degree_rank1", degree1_rank)
                print("degree_rank2", degree2_rank)
                print("company_rank", company_rank)
                print("experience", experience)
                print("notice_period", notice_period)
                print("exp_ctc", exp_ctc)

                valid_indices = [i for i, rank in enumerate(college_rank) if rank > 0]
                # Check and get index of minimum value if available
                min_index = min(valid_indices, key=lambda i: college_rank[i]) if valid_indices else None

                if min_index == None:
                    selected_clg = college[0]
                    selected_rank = college_rank[0]
                    print("XXX", selected_clg, selected_rank)
                else:
                    selected_clg = college[min_index]
                    selected_rank = college_rank[min_index]
                    print("XXX", selected_clg, selected_rank)

                degree_str = ", ".join(degree)
                company_str = ", ".join(company)

                # Determine the total based on ranking source
                ranking_sources = total[1]
                total_ranks = []
                for source in ranking_sources:
                    if source.upper() == 'QS':
                        total_ranks.append(total_qs)
                    else:
                        total_ranks.append(total_nirf)

                # Use correct total for each rank in score calculation
                college1_total = total_ranks[0] if len(total_ranks) > 0 else total_nirf
                college2_total = total_ranks[1] if len(total_ranks) > 1 else total_nirf
                print("college1_total",college1_total)
                print("college2_total",college2_total)
                score = (
                    ((1 - min((college1_rank - 1) / college1_total, 1)) * 0.30)
                    + (company_rank * 0.15)
                    + ((1 - min((college2_rank - 1) / college2_total, 1)) * 0.10)
                    + (degree1_rank * 0.10)
                    + (min(experience / 10, 1) * 0.10)
                    + (degree2_rank * 0.05)
                    + (min(notice_period / 90, 1) * 0.05)
                    + ((1 - exp_ctc / 50) * 0.05)
                )

                score = score*100
                print(score)
                score = ((math.ceil(score))/100)
                print("final_score", score)

                if college1_rank >= 1 or college2_rank >= 1:
                    explain = (
                    f"{name} has graduated from {selected_clg}, which holds a {ranking[0]} rank of {selected_rank}. "
                    f"Candidate holds a degree in {degree_str} with {experience} years of experience, "
                    f"and have worked at companies like {company_str}.\n"
                    f"The expected CTC is ₹{exp_ctc_1}, and the notice period is {notice_period} days.\n"
                    f"The final score assigned to this resume is {score}.\n\n")
                    print(explain)
                else:
                    explain = (
                        f"{name} has graduated from {selected_clg} and which is not reputed institute. "
                        f"Candidate holds a degree in {degree_str} with {experience} years of experience, "
                        f"and have worked at companies like {company_str}.\n"
                        f"The expected CTC is ₹{exp_ctc_1}, and the notice period is {notice_period} days.\n"
                        f"The final score assigned to this resume is {score}.\n\n")
                    print(explain)

###################### for summarizing resume explainalibility ###################################################
                def summarize_resume(explain):
                    prompt = (
                    "Here is a resume explanation:\n\n"
                    f"{explain}\n\n"
                    "Task:\n"
                    "Write a user-friendly, two-line summary of this candidate. "
                    "Keep it short, simple, and easy to understand—avoid technical jargon or score formulas. "
                    "Clearly mention the candidate's education background and the companies they have worked at. "
                    "Also include the final score assigned to this resume. "
                    "If the candidate scored very well (above 78%), avoid using negative terms like 'low'. "
                    "Instead, mention gentle reasons such as college not being in the NIRF or QS ranking, less experience, or not having worked in reputed companies. "
                    "Briefly justify the score based on factors like top-ranked college, reputed companies, or overall experience."
                    )

                    raw_output = invoke_gpt4(prompt)
                    # print("LLM Raw Output:", raw_output)
                    return raw_output

                summarized_response = summarize_resume(explain)+"\n\n"+result
                print("ZZZ", summarized_response)

####################################################################################################
                resume_url = f"{base_url}uploaded_resumes/{filename}"
                explainability = f"{base_url}resume-details?{urlencode({'explanation': summarized_response})}"
                
                new_results.append(
                    {
                        "name": name,
                        "email": email,
                        "processed_date":processed_date,
                        "phone_number": phone_number,
                        "college": college,
                        "degree": degree,
                        "company": company,
                        "skills": skills,
                        "experience": experience,
                        "score": score,
                        "file_hash": file_hash,
                        "resume_link": resume_url,
                        "explainability": explainability
                    }
                )
        except Exception as e:
            print("Resume processing error:", e)


    if new_results:
        try:
            file_exists = os.path.exists(CSV_OUTPUT_PATH)
            with open(CSV_OUTPUT_PATH, "a", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "name",
                    "email",
                    "processed_date",
                    "phone_number",
                    "college",
                    "degree",
                    "company",
                    "skills",
                    "experience",
                    "score",
                    "file_hash",
                    "resume_link",
                    "explainability"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(new_results)
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    all_results = load_all_records(CSV_OUTPUT_PATH)

    return JSONResponse(
        content={
            "processed_count": len(new_results),
            "duplicate_count": len(duplicates),
            "new_results": [
                {
                    "name": r["name"],
                    "email": r["email"],
                    "processed_date": r["processed_date"],
                    "phone_number": r["phone_number"],
                    "score": r["score"],
                    "resume_link": r["resume_link"],
                    "explainability": r["explainability"]
                }
                for r in new_results
            ],
            "duplicates": duplicates,
            "all_results": all_results,
            "csv_path": CSV_OUTPUT_PATH if all_results else None,
            "message": "Upload finished.",
        }
    )

################################Remove Function#########################################
# REMOVED_RESUMES_PATH = r"Removed_Resumes/Removed_Resumes_Data.csv"

# @app.delete("/remove-resume/")
# def remove_resume(
#     identifier: str = Query(..., description="Enter phone number or email"),
#     removal_reason: str = Body(..., embed=True, description="Reason for removal")
# ):
#     if not os.path.exists(CSV_OUTPUT_PATH):
#         raise HTTPException(status_code=404, detail="Resume data not found")

#     if not identifier:
#         raise HTTPException(status_code=400, detail="Please provide a phone number or email")

#     # Email check
#     is_email = bool(re.match(r"^[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$", identifier))

#     # Read CSV
#     with open(CSV_OUTPUT_PATH, newline='', encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         rows = list(reader)
#         fieldnames = reader.fieldnames

#     # Match based on identifier
#     if is_email:
#         matched_rows = [row for row in rows if row.get("email") == identifier]
#         filtered_rows = [row for row in rows if row.get("email") != identifier]
#         field_used = "email"
#     else:
#         matched_rows = [row for row in rows if row.get("phone_number") == identifier]
#         filtered_rows = [row for row in rows if row.get("phone_number") != identifier]
#         field_used = "phone_number"

#     if not matched_rows:
#         raise HTTPException(status_code=404, detail=f"No resume found with {field_used}: {identifier}")

#     # Write back filtered data
#     with open(CSV_OUTPUT_PATH, "w", newline='', encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(filtered_rows)

#     # Prepare removed data with reason
#     removed_data_fields = ["name", "email", "phone_number", "resume_link","removal_reason"]
#     to_store = []
#     for row in matched_rows:
#         to_store.append({
#             "name": row.get("name", "Not Available"),
#             "email": row.get("email", "Not Available"),
#             "phone_number": row.get("phone_number", "Not Available"),
#             "resume_link": row.get("resume_link", "Not Available"),
#             "removal_reason": removal_reason
#         })

#     # Write to Removed_Resumes_Data.csv
#     file_exists = os.path.exists(REMOVED_RESUMES_PATH)
#     with open(REMOVED_RESUMES_PATH, "a", newline='', encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=removed_data_fields)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerows(to_store)

#     return {"message": f"Resume with {field_used} '{identifier}' has been removed and logged."}
# #####################################################################################################
REMOVED_RESUMES_PATH = r"Removed_Resumes/Removed_Resumes_Data.csv"

@app.delete("/remove-resume/")
def remove_resume(
    identifier: str = Query(..., description="Enter phone number or email"),
    removal_reason: str = Body(..., embed=True, description="Reason for removal")
):
    if not os.path.exists(CSV_OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="Resume data not found")

    if not identifier:
        raise HTTPException(status_code=400, detail="Please provide a phone number or email")

    # Email check
    is_email = bool(re.match(r"^[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$", identifier))

    # Read CSV
    with open(CSV_OUTPUT_PATH, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Match based on identifier
    if is_email:
        matched_rows = [row for row in rows if row.get("email") == identifier]
        filtered_rows = [row for row in rows if row.get("email") != identifier]
        field_used = "email"
    else:
        matched_rows = [row for row in rows if row.get("phone_number") == identifier]
        filtered_rows = [row for row in rows if row.get("phone_number") != identifier]
        field_used = "phone_number"

    if not matched_rows:
        raise HTTPException(status_code=404, detail=f"No resume found with {field_used}: {identifier}")

    # Write back filtered data
    with open(CSV_OUTPUT_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    # Prepare removed data with reason
    removed_data_fields = ["name", "email", "phone_number", "resume_link", "removal_reason"]
    to_store = []
    for row in matched_rows:
        to_store.append({
            "name": row.get("name", "Not Available"),
            "email": row.get("email", "Not Available"),
            "phone_number": row.get("phone_number", "Not Available"),
            "resume_link": row.get("resume_link", "Not Available"),
            "removal_reason": removal_reason
        })

    # Ensure directory exists
    os.makedirs(os.path.dirname(REMOVED_RESUMES_PATH), exist_ok=True)

    # Write to Removed_Resumes_Data.csv
    file_exists = os.path.exists(REMOVED_RESUMES_PATH)
    with open(REMOVED_RESUMES_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=removed_data_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerows(to_store)

    return {"message": f"Resume with {field_used} '{identifier}' has been removed and logged."}
