import re
from typing import List
import pandas as pd
from rapidfuzz import process


#  Utility Functions 
def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', text)).strip().lower()

def normalize_with_alias(name, alias_dict):
    cleaned = clean_text(name)
    for alias, standard in alias_dict.items():
        if clean_text(alias) == cleaned:
            return standard
    return name

def fuzzy_match_exists(name, reference_dict, aliases_dict, threshold=95):
    name = normalize_with_alias(name, aliases_dict)
    cleaned_name = clean_text(name)
    match, score, _ = process.extractOne(cleaned_name, reference_dict.keys())
    return 1 if score >= threshold else 0.5

#Load Company Data 
company_df = pd.read_csv(r"Data\company_rankings.csv")
company_rank_dict = {clean_text(name): 1 for name in company_df['company_name']}  

company_aliases = {
    "TCS": "Tata Consultancy Services",
    "Tata": "Tata Consultancy Services",
    "TCS Ltd": "Tata Consultancy Services",
    "Infosys Ltd": "Infosys",
    "Infosys Limited": "Infosys",
    "Wipro Ltd": "Wipro",
    "Wipro Limited": "Wipro",
    "Accenture": "Accenture plc",
    "Accenture Ltd": "Accenture plc",
    "Accenture Limited": "Accenture plc",
    "Cognizant": "Cognizant Technology Solutions",
    "CTS": "Cognizant Technology Solutions",
    "Tech Mahindra Ltd": "Tech Mahindra",
    "TechM": "Tech Mahindra",
    "IBM India": "IBM",
    "HCL Tech": "HCL Technologies",
    "HCL Technologies Ltd": "HCL Technologies",
    "Capgemini India": "Capgemini",
    "LTI": "LTIMindtree",
    "Mindtree": "LTIMindtree",
    "Google India": "Google",
    "Amazon Web Services": "Amazon",
    "AWS": "Amazon",
    "Facebook": "Meta",
    "Meta Platforms Inc.": "Meta",
    "Ernst & Young": "EY",
    "EY India": "EY",
    "PricewaterhouseCoopers": "PwC",
    "PwC India": "PwC",
    "KPMG India": "KPMG",
    "Deloitte India": "Deloitte"
}

def get_company_ranks(company_list):
    return [fuzzy_match_exists(company, company_rank_dict, company_aliases) for company in company_list]
#-------------------------------------------------------------------------------------------------------------------
# Utility Functions
def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', text)).strip().lower()

def normalize_with_alias(name, alias_dict):
    cleaned = clean_text(name)
    for alias, standard in alias_dict.items():
        if clean_text(alias) == cleaned:
            return standard
    return name

def fuzzy_match(name, reference_dict, aliases_dict, threshold=90):
    name = normalize_with_alias(name, aliases_dict)
    cleaned_name = clean_text(name)
    match, score, _ = process.extractOne(cleaned_name, reference_dict.keys())
    if score >= threshold:
        return reference_dict[match]
    return 0

#  Load Data 
nirf_df = pd.read_csv(r"Data\nirf_2024.csv")
college_rank_dict = {clean_text(name): rank for name, rank in zip(nirf_df['college_name'], nirf_df['rank'])}

# -------- College Aliases --------
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

# College Rank Function 
def get_college_ranks(college_list):
    ranks = [fuzzy_match(college, college_rank_dict, college_aliases) for college in college_list]
    return ranks


_PATTERNS = [
    r"\bcomputer(\s*science(s)?|\s*engineering)?\b",
    r"\bcs\b",
    r"\binformation\s*(technology|science)?\b",
    r"\bit\b",
    r"\bis\b",
    r"\belectrical(\s*(and|&)?\s*electronics)?\s*engineering\b",
    r"\bee\b",
    r"\beee\b",
    r"\belectronics\s*(and|&)?\s*communication\b",
    r"\bec\b",
    r"\bece\b",  
    r"\belectronics\b",
    r"\bmath(ematics)?\b",
    r"\bdata\s*sciences?\b",
    r"\bscience\b",
    r"\bai\b",
    r"\bartificial\s*intelligence\b",
]

def get_degree_ranks(degrees: List[str]) -> List[float]:
    """Return 1.0 if a degree mentions CS/IT/EE/EC/Math, else 0.5."""
    ranks = []
    for degree in degrees:
        text = degree.lower()
        score = 0.5                    
        for pat in _PATTERNS:
            if re.search(pat, text):
                score = 1.0
                break
        ranks.append(score)
    return ranks
