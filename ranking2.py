import os
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from rapidfuzz import process

logging.basicConfig(level=logging.INFO)

load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


college_aliases = {
    "IIT Madras": "Indian Institute of Technology Madras",
    "IIT Bombay": "Indian Institute of Technology Bombay",
    "IIT Delhi": "Indian Institute of Technology Delhi",
    "IIT Kanpur": "Indian Institute of Technology Kanpur",
    "IIT Kharagpur": "Indian Institute of Technology Kharagpur",
    "IIT Roorkee": "Indian Institute of Technology Roorkee",
    "IIT Guwahati": "Indian Institute of Technology Guwahati",
    "IIT Hyderabad": "Indian Institute of Technology Hyderabad",
    "IIT BHU": "Indian Institute of Technology (Banaras Hindu University) Varanasi",
    "IIT Varanasi": "Indian Institute of Technology (Banaras Hindu University) Varanasi",
    "IIT Gandhinagar": "Indian Institute of Technology Gandhinagar",
    "IIT Indore": "Indian Institute of Technology Indore",
    "IIT Dhanbad": "Indian Institute of Technology (Indian School of Mines) Dhanbad",
    "BITS": "Birla Institute of Technology and Science",
    "BITS Pilani": "Birla Institute of Technology and Science (BITS Pilani)",
    "BITS Hyderabad": "Birla Institute of Technology and Science (BITS Pilani)",
    "BITS Goa": "Birla Institute of Technology and Science (BITS Pilani)",
    "IISc": "Indian Institute of Science, Bengaluru",
    "JNU": "Jawaharlal Nehru University",
    "VIT": "Vellore Institute of Technology",
    "SRM": "S.R.M. Institute of Science and Technology",
    "Amrita University": "Amrita Vishwa Vidyapeetham",
    "Manipal University": "Manipal Academy of Higher Education, Manipal",
    "DU": "University of Delhi",
    "Anna Univ": "Anna University",
    "JMI": "Jamia Millia Islamia",
    "AMU": "Aligarh Muslim University",
    "CU": "Calcutta University",
    "PU": "Panjab University",
    "DTU": "Delhi Technological University",
    "NIT Trichy": "National Institute of Technology Tiruchirappalli",
    "NIT Surathkal": "National Institute of Technology Karnataka, Surathkal",
    "NIT Rourkela": "National Institute of Technology Rourkela",
    "NIT Warangal": "National Institute of Technology Warangal",
    "NIT Calicut": "National Institute of Technology Calicut",
    "NIT Jalandhar": "Dr. B R Ambedkar National Institute of Technology Jalandhar",
    "NIT Silchar": "National Institute of Technology Silchar",
    "NIT Durgapur": "National Institute of Technology Durgapur",
    "NIT Kurukshetra": "National Institute of Technology Kurukshetra",
    "IIIT Hyderabad": "International Institute of Information Technology Hyderabad"
}


def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', text)).strip().lower()

def normalize_with_alias(name, alias_dict):
    cleaned = clean_text(name)
    for alias, standard in alias_dict.items():
        if clean_text(alias) == cleaned:
            return standard
    return name

def invoke_gpt4(prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            temperature=0,
            max_tokens=10,
            messages=[
                {"role": "system", "content": "Classify whether a college is located in India or outside India. Reply only with '1' or '0'."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API failed: {e}")
        return "0"

def is_indian(college: str) -> bool:
    prompt = f'Is the college "{college}" located in India? Reply only "1" or "0".'
    match = re.search(r"\b[01]\b", invoke_gpt4(prompt))
    return bool(match and match.group() == "1")


def load_nirf_data(path: str) -> dict:
    df = pd.read_csv(path)
    return {clean_text(name): str(rank).strip() for name, rank in zip(df["college_name"], df["rank"])}

def load_qs_data(path: str) -> dict:
    df = pd.read_csv(path)
    return {clean_text(name): int(rank) for rank, name in zip(df["Rank"], df["College name"])}

def parse_rank(rank_str: str) -> int:
    if pd.isna(rank_str) or not str(rank_str).strip():
        return 0
    match = re.match(r"(\d+)", str(rank_str))
    return int(match.group(1)) if match else 0


def fuzzy_find_rank(college: str, rank_dict: dict, source: str, threshold=95) -> int:
    college_normalized = clean_text(normalize_with_alias(college, college_aliases))
    match, score, _ = process.extractOne(college_normalized, rank_dict.keys())
    if score >= threshold:
        rank = parse_rank(rank_dict[match])
        logging.info(f"{college} → Rank: {rank} (Fuzzy Match: {match}, Score: {score}, Source: {source})")
        return rank
    logging.info(f"{college} → Rank: 0 (Not found in {source})")
    return 0


def get_college_ranks(college_list: list[str]) -> list[int]:
    nirf_path = r"DATA\nirf_2024.csv"
    qs_path = r"DATA\QS Ranking.csv"

    nirf_data = load_nirf_data(nirf_path)
    qs_data = load_qs_data(qs_path)

    ranks = []
    sources = []
    for college in college_list:
        try:
            if is_indian(college):
                rank = fuzzy_find_rank(college, nirf_data, "NIRF")
                source = "NIRF"
            else:
                rank = fuzzy_find_rank(college, qs_data, "QS")
                source = "QS"
            ranks.append(rank)
            sources.append(source)
        except Exception as e:
            logging.error(f"Failed to evaluate {college}: {e}")
            ranks.append(0)
            sources.append("Unknown")
    return ranks,sources


# if __name__ == "__main__":
#     input_colleges = [
#         "Amity University",
#         "University of Oxford",
#         "Don Bosco Institute of Technology",
#         "IIT Bombay",
#         "Indian Institute of Technology Bombay",
#         "Harvard University",
#         "NIT Trichy",
#         "Mumbai University",
#         "aMITY",
#         "Northeastern",
#         "Kalinga Institute of Industrial Technology, Bhubaneswar, India",
#         "Rutgers University, New Brunswick",
#         "Awdfsc"
#     ]
#     results = get_college_ranks(input_colleges)
#     print("Final Ranks:", results)
