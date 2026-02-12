import os
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from google import genai  # Gemini generate_content pattern [web:61]

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Medical Diagnosis Assistant (Capstone)", version="1.0.0")

# Safe static mount (won't crash if folder missing)
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=True), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DISCLAIMER = (
    "Disclaimer: Educational clinical decision support only. "
    "Not a medical diagnosis and not medical advice."
)

# ---------- Module 1: ICD suggestion (demo) ----------
ICD10_DEMO = [
    {"code": "J10.1", "title": "Influenza with other respiratory manifestations", "keywords": ["flu", "influenza", "fever", "cough", "myalgia"]},
    {"code": "J00", "title": "Acute nasopharyngitis [common cold]", "keywords": ["cold", "coryza", "sneezing", "sore", "throat"]},
    {"code": "A09", "title": "Infectious gastroenteritis and colitis, unspecified", "keywords": ["diarrhea", "vomiting", "gastro", "abdominal", "pain"]},
    {"code": "B34.9", "title": "Viral infection, unspecified", "keywords": ["viral", "fever", "malaise"]},
]

def icd_rank(symptoms: str):
    toks = set(symptoms.lower().replace(",", " ").split())
    out = []
    for row in ICD10_DEMO:
        keys = set(row["keywords"])
        matched = sorted(list(toks.intersection(keys)))
        score = len(matched) / max(1, len(keys))
        if score > 0:
            out.append({"code": row["code"], "title": row["title"], "score": round(score, 3), "matched": matched})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:5]

# ---------- Module 2: Immuno profiling (proxy) ----------
def immuno_profile_proxy(symptoms: str):
    t = symptoms.lower()
    inflammation = 0.2
    if any(w in t for w in ["fever", "chills"]):
        inflammation += 0.3
    if any(w in t for w in ["rash", "hives", "allergy", "itch"]):
        inflammation += 0.3
    if any(w in t for w in ["diarrhea", "vomiting"]):
        inflammation += 0.2
    inflammation = min(1.0, round(inflammation, 2))

    axis = "Innate (acute)"
    if "allergy" in t or "hives" in t:
        axis = "Th2-skewed (allergic)"
    elif "autoimmune" in t:
        axis = "Autoimmune-like"

    return {
        "immune_axis": axis,
        "inflammation_score": inflammation,
        "notes": "Proxy profiling from symptom text. Replace with your immunoinformatics pipeline.",
    }

# ---------- Module 3: Gemini AI (Flash Preview) ----------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
gemini_key = os.getenv("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None

def gemini_explain(symptoms: str, icd_candidates, immuno_profile) -> str:
    if gemini_client is None:
        return "GEMINI_API_KEY not set in Render Environment."

    prompt = f"""
You are an academic clinical decision-support assistant.
Do NOT claim to diagnose.

{DISCLAIMER}

Symptoms:
{symptoms}

ICD-10 candidates:
{icd_candidates}

Immuno profile:
{immuno_profile}

Write:
1) Why these are candidates (no diagnosis).
2) 3 follow-up questions.
3) 3 generic red flags that need urgent care.
Keep it concise.
""".strip()

    resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)  # [web:61]
    return resp.text or ""

# ---------- Module 4: Report ----------
def build_report(symptoms: str, icd_candidates, immuno_profile, ai_text: str) -> str:
    return f"""IMDS CAPSTONE REPORT (Demo)

{DISCLAIMER}

Symptoms:
{symptoms}

ICD-10 Candidates:
{icd_candidates}

Immunoinformatics Profiling (Proxy):
{immuno_profile}

Gemini AI Explanation:
{ai_text}
"""

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/module/icd", response_class=HTMLResponse)
def module1_get(request: Request):
    return templates.TemplateResponse("module1_icd.html", {"request": request, "results": None, "symptoms": ""})

@app.post("/module/icd", response_class=HTMLResponse)
def module1_post(request: Request, symptoms: str = Form(...)):
    results = icd_rank(symptoms)
    return templates.TemplateResponse("module1_icd.html", {"request": request, "results": results, "symptoms": symptoms})

@app.get("/module/immuno", response_class=HTMLResponse)
def module2_get(request: Request):
    return templates.TemplateResponse("module2_immuno.html", {"request": request, "profile": None, "symptoms": ""})

@app.post("/module/immuno", response_class=HTMLResponse)
def module2_post(request: Request, symptoms: str = Form(...)):
    profile = immuno_profile_proxy(symptoms)
    return templates.TemplateResponse("module2_immuno.html", {"request": request, "profile": profile, "symptoms": symptoms})

@app.get("/module/ai", response_class=HTMLResponse)
def module3_get(request: Request):
    return templates.TemplateResponse("module3_ai.html", {"request": request, "ai_text": None, "symptoms": ""})

@app.post("/module/ai", response_class=HTMLResponse)
def module3_post(request: Request, symptoms: str = Form(...)):
    cands = icd_rank(symptoms)
    prof = immuno_profile_proxy(symptoms)
    ai_text = gemini_explain(symptoms, cands, prof)
    return templates.TemplateResponse("module3_ai.html", {"request": request, "ai_text": ai_text, "symptoms": symptoms})

@app.get("/module/reports", response_class=HTMLResponse)
def module4_get(request: Request):
    return templates.TemplateResponse("module4_reports.html", {"request": request, "report": None, "symptoms": ""})

@app.post("/module/reports", response_class=HTMLResponse)
def module4_post(request: Request, symptoms: str = Form(...)):
    cands = icd_rank(symptoms)
    prof = immuno_profile_proxy(symptoms)
    ai_text = gemini_explain(symptoms, cands, prof)
    report = build_report(symptoms, cands, prof, ai_text)
    return templates.TemplateResponse("module4_reports.html", {"request": request, "report": report, "symptoms": symptoms})
