import os
from typing import Optional, List

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from itsdangerous import URLSafeSerializer, BadSignature

from pydantic import BaseModel, Field
from openai import OpenAI

# ---------------- App setup ----------------
app = FastAPI(title="IMDS Capstone Web", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")  # standard FastAPI templating [web:51]

# ---------------- Security (simple cookie session) ----------------
# For capstone demo. For real production use JWT + hashed passwords + database.
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-render")
serializer = URLSafeSerializer(SECRET_KEY, salt="imds-session")

DEMO_USER = os.getenv("DEMO_USER", "student")
DEMO_PASS = os.getenv("DEMO_PASS", "student123")

SESSION_COOKIE = "imds_session"

def set_session(response: RedirectResponse, username: str):
    token = serializer.dumps({"u": username})
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # Render uses HTTPS at the edge; you can set True later if needed
        max_age=60 * 60 * 8,
    )

def clear_session(response: RedirectResponse):
    response.delete_cookie(SESSION_COOKIE)

def get_user_from_request(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        data = serializer.loads(token)
        return data.get("u")
    except BadSignature:
        return None

def require_login(request: Request) -> Optional[RedirectResponse]:
    if not get_user_from_request(request):
        return RedirectResponse(url="/login", status_code=303)
    return None

# ---------------- Your 4 modules (minimal working placeholders) ----------------
# Module 1: ICD suggestion (demo)
ICD10_DEMO = [
    {"code": "J10.1", "title": "Influenza with other respiratory manifestations", "keywords": ["flu", "influenza", "fever", "cough", "myalgia"]},
    {"code": "J00", "title": "Acute nasopharyngitis [common cold]", "keywords": ["cold", "coryza", "sneezing", "sore", "throat"]},
    {"code": "A09", "title": "Infectious gastroenteritis and colitis, unspecified", "keywords": ["diarrhea", "vomiting", "gastro", "abdominal", "pain"]},
]

def simple_rank(symptoms: str):
    toks = set(symptoms.lower().replace(",", " ").split())
    results = []
    for row in ICD10_DEMO:
        keys = set(row["keywords"])
        matched = sorted(list(toks.intersection(keys)))
        score = len(matched) / max(1, len(keys))
        if score > 0:
            results.append({"code": row["code"], "title": row["title"], "score": round(score, 3), "matched": matched})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:5]

# Module 2: Immuno profiling (proxy)
def proxy_immuno(symptoms: str):
    t = symptoms.lower()
    inflammation = 0.2
    if "fever" in t:
        inflammation += 0.3
    if "rash" in t or "hives" in t or "allergy" in t:
        inflammation += 0.3
    if "diarrhea" in t or "vomiting" in t:
        inflammation += 0.2
    inflammation = min(1.0, round(inflammation, 2))

    axis = "Innate (acute)"
    if "allergy" in t or "hives" in t:
        axis = "Th2-skewed (allergic)"
    return {"immune_axis": axis, "inflammation_score": inflammation}

# Module 3: AI narrative (LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Module 4: Reports (placeholder)
def make_report(symptoms: str, icd, immuno, ai_text: str):
    return f"""IMDS CAPSTONE REPORT (Demo)
Symptoms: {symptoms}

ICD Candidates:
{icd}

Immuno Profile:
{immuno}

AI Narrative:
{ai_text}
"""

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    if get_user_from_request(request):
        return RedirectResponse(url="/dashboard", status_code=303)
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
def login_post(username: str = Form(...), password: str = Form(...)):
    if username == DEMO_USER and password == DEMO_PASS:
        resp = RedirectResponse(url="/dashboard", status_code=303)
        set_session(resp, username=username)
        return resp
    return templates.TemplateResponse("login.html", {"request": Request, "error": "Invalid username/password"}, status_code=401)

@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    clear_session(resp)
    return resp

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    gate = require_login(request)
    if gate:
        return gate
    user = get_user_from_request(request)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    gate = require_login(request)
    if gate:
        return gate
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/module/icd", response_class=HTMLResponse)
def module1_get(request: Request):
    gate = require_login(request)
    if gate:
        return gate
    return templates.TemplateResponse("module1_icd.html", {"request": request, "results": None})

@app.post("/module/icd", response_class=HTMLResponse)
def module1_post(request: Request, symptoms: str = Form(...)):
    gate = require_login(request)
    if gate:
        return gate
    results = simple_rank(symptoms)
    return templates.TemplateResponse("module1_icd.html", {"request": request, "results": results, "symptoms": symptoms})

@app.get("/module/immuno", response_class=HTMLResponse)
def module2_get(request: Request):
    gate = require_login(request)
    if gate:
        return gate
    return templates.TemplateResponse("module2_immuno.html", {"request": request, "profile": None})

@app.post("/module/immuno", response_class=HTMLResponse)
def module2_post(request: Request, symptoms: str = Form(...)):
    gate = require_login(request)
    if gate:
        return gate
    profile = proxy_immuno(symptoms)
    return templates.TemplateResponse("module2_immuno.html", {"request": request, "profile": profile, "symptoms": symptoms})

@app.get("/module/ai", response_class=HTMLResponse)
def module3_get(request: Request):
    gate = require_login(request)
    if gate:
        return gate
    return templates.TemplateResponse("module3_ai.html", {"request": request, "ai_text": None})

@app.post("/module/ai", response_class=HTMLResponse)
def module3_post(request: Request, symptoms: str = Form(...)):
    gate = require_login(request)
    if gate:
        return gate

    if client is None:
        ai_text = "OPENAI_API_KEY not set. Add it in Render → Environment as a secret."
        return templates.TemplateResponse("module3_ai.html", {"request": request, "ai_text": ai_text, "symptoms": symptoms})

    cands = simple_rank(symptoms)
    prof = proxy_immuno(symptoms)

    system = (
        "You are an academic clinical decision-support assistant. "
        "Do NOT claim to diagnose. Provide cautious explanation, suggest follow-up questions and red flags."
    )

    user_msg = f"""
Symptoms:
{symptoms}

ICD candidates (demo):
{cands}

Immuno profile (proxy):
{prof}

Write a concise decision-support narrative (no diagnosis).
""".strip()

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    ai_text = resp.choices[0].message.content or ""
    return templates.TemplateResponse("module3_ai.html", {"request": request, "ai_text": ai_text, "symptoms": symptoms})

@app.get("/module/reports", response_class=HTMLResponse)
def module4_get(request: Request):
    gate = require_login(request)
    if gate:
        return gate
    return templates.TemplateResponse("module4_reports.html", {"request": request, "report": None})

@app.post("/module/reports", response_class=HTMLResponse)
def module4_post(request: Request, symptoms: str = Form(...)):
    gate = require_login(request)
    if gate:
        return gate

    icd = simple_rank(symptoms)
    immuno = proxy_immuno(symptoms)
    ai_text = "(Generate from AI module, or integrate here.)"

    report = make_report(symptoms, icd, immuno, ai_text)
    return templates.TemplateResponse("module4_reports.html", {"request": request, "report": report, "symptoms": symptoms})

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
