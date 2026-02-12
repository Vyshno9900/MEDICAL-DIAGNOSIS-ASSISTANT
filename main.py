import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from itsdangerous import URLSafeSerializer, BadSignature

from google import genai  # Gemini generate_content docs pattern [web:61]


# ---------------- App setup ----------------
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="IMDS Capstone Web", version="1.0.0")

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=True), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DISCLAIMER = (
    "Disclaimer: Educational clinical decision support only. "
    "Not a medical diagnosis and not medical advice."
)

# ---------------- Login (cookie session) ----------------
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-render")
serializer = URLSafeSerializer(SECRET_KEY, salt="imds-session")

DEMO_USER = os.getenv("DEMO_USER", "student")
DEMO_PASS = os.getenv("DEMO_PASS", "student123")
SESSION_COOKIE = "imds_session"

# If you later use a separate frontend domain, cookie settings must change.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"  # set to 1 after everything works
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")   # lax is fine for same-site
COOKIE_PATH = "/"

def set_session_cookie(resp: RedirectResponse, username: str):
    token = serializer.dumps({"u": username})
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        path=COOKIE_PATH,
        max_age=60 * 60 * 8,
    )

def clear_session_cookie(resp: RedirectResponse):
    resp.delete_cookie(key=SESSION_COOKIE, path=COOKIE_PATH)

def get_user(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        data = serializer.loads(token)
        return data.get("u")
    except BadSignature:
        return None

def login_required(request: Request) -> Optional[RedirectResponse]:
    if not get_user(request):
        return RedirectResponse("/login", status_code=303)
    return None


# ---------------- Module 1: ICD candidate suggestion (demo dataset) ----------------
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


# ---------------- Module 2: Immunoinformatics profiling (proxy) ----------------
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


# ---------------- Module 3: Gemini AI explanation ----------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")
gemini_key = os.getenv("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None

def gemini_explain(symptoms: str, icd_candidates, immuno_profile) -> str:
    if gemini_client is None:
        return "GEMINI_API_KEY is not set. Add it in Render → Environment."

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


# ---------------- Module 4: Report generator ----------------
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


# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return RedirectResponse("/dashboard" if get_user(request) else "/login", status_code=303)

@app.get("/login
