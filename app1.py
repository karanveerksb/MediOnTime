# app.py (Backend)
import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import urllib3
import dotenv
from fastapi import FastAPI, HTTPException, WebSocket, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from apscheduler.schedulers.background import BackgroundScheduler
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.base.exceptions import TwilioRestException
import google.generativeai as genai

# Configuration
dotenv.load_dotenv(verbose=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger("MedicationReminder")

# Rate Limiting
class RateLimiter:
    def __init__(self, rate: int, bucket_size: int):
        self.rate = rate
        self.tokens = bucket_size
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def get_token(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            refill = elapsed * (self.rate / 60)
            self.tokens = min(self.tokens + refill, self.rate)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

gemini_limiter = RateLimiter(60, 30)

# Gemini Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
)

# Twilio Setup
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

# Application State
scheduler = BackgroundScheduler()
active_calls = {}
medication_schedules = {}
adherence_logs = {}
websocket_connections = {}

# Models
class MedicationRequest(BaseModel):
    patient: str
    medication: str
    dosage: str
    schedule: Dict[str, str]
    phone: str
    system_message: str

    @validator('phone')
    def validate_phone(cls, v):
        if not v.startswith('+'): raise ValueError('Invalid phone format')
        return v

    @validator('schedule')
    def validate_schedule(cls, v):
        for t in v.values(): datetime.strptime(t, "%H:%M")
        return v

# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/schedule")
async def schedule_medication(request: MedicationRequest):
    try:
        # Schedule implementation from previous answer
        # ...
        return {"status": "scheduled"}
    except Exception as e:
        logger.error(f"Scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    await websocket.accept()
    if phone not in websocket_connections:
        websocket_connections[phone] = []
    websocket_connections[phone].append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections[phone].remove(websocket)

# ... (Other backend endpoints from previous answer)

# streamlit_app.py (Frontend)
import streamlit as st
import requests
import json
import websocket
import threading
import time
from datetime import datetime

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws/")

# Session State
if 'ws' not in st.session_state:
    st.session_state.ws = None
if 'adherence' not in st.session_state:
    st.session_state.adherence = []
if 'calls' not in st.session_state:
    st.session_state.calls = []

class WebSocketClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.running = False
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        
    def run(self):
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(self.url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close)
                self.ws.run_forever()
            except Exception as e:
                st.error(f"WebSocket error: {str(e)}")
            time.sleep(5)
            
    def on_message(self, ws, message):
        data = json.loads(message)
        if data['type'] == 'adherence_update':
            st.session_state.adherence.append(data['data'])
        elif data['type'] == 'call_update':
            st.session_state.calls = data['data']
            
    def on_error(self, ws, error):
        st.error(f"WebSocket error: {str(error)}")
        
    def on_close(self, ws, close_status_code, close_msg):
        st.warning("WebSocket connection closed")

def connect_websocket(phone):
    if st.session_state.ws:
        st.session_state.ws.running = False
    st.session_state.ws = WebSocketClient(f"{WEBSOCKET_URL}{phone}")
    st.session_state.ws.start()

def schedule_medication():
    try:
        response = requests.post(f"{API_URL}/schedule", json={
            "patient": st.session_state.patient_name,
            "medication": st.session_state.medication,
            "dosage": st.session_state.dosage,
            "schedule": st.session_state.schedule,
            "phone": st.session_state.phone,
            "system_message": f"Assistant for {st.session_state.patient_name}"
        })
        if response.status_code == 200:
            connect_websocket(st.session_state.phone)
            st.success("Medication scheduled!")
        else:
            st.error(response.text)
    except Exception as e:
        st.error(str(e))

# Streamlit UI
st.title("Medication Adherence System")

with st.sidebar:
    st.header("Patient Setup")
    phone = st.text_input("Phone Number (+1234567890)", key="phone")
    patient_name = st.text_input("Patient Name", key="patient_name")
    medication = st.text_input("Medication Name", key="medication")
    dosage = st.text_input("Dosage", key="dosage")
    
    st.header("Dosage Schedule")
    schedule = {}
    if st.checkbox("Morning Dose"):
        schedule["morning"] = st.time_input("Morning Time").strftime("%H:%M")
    if st.checkbox("Afternoon Dose"):
        schedule["afternoon"] = st.time_input("Afternoon Time").strftime("%H:%M")
    if st.checkbox("Evening Dose"):
        schedule["evening"] = st.time_input("Evening Time").strftime("%H:%M")
        
    if st.button("Schedule Medication"):
        st.session_state.schedule = schedule
        schedule_medication()

# Main Interface
tab1, tab2, tab3 = st.tabs(["Adherence", "Calls", "Dashboard"])

with tab1:
    st.header("Adherence Tracking")
    if st.button("Refresh Data"):
        try:
            response = requests.get(f"{API_URL}/adherence/{phone}")
            st.session_state.adherence = response.json()
        except:
            st.error("Could not fetch adherence data")
            
    for entry in st.session_state.adherence:
        st.write(f"{entry['timestamp']} - {entry['medication']} ({entry['status']})")

with tab2:
    st.header("Call Management")
    if st.button("Refresh Calls"):
        try:
            response = requests.get(f"{API_URL}/calls")
            st.session_state.calls = response.json()
        except:
            st.error("Could not fetch calls")
            
    for call in st.session_state.calls:
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"Call to {call['phone']} - {call['status']}")
        with col2:
            if st.button(f"Cancel {call['call_sid']}"):
                requests.delete(f"{API_URL}/calls/{call['call_sid']}")

with tab3:
    st.header("Patient Dashboard")
    if phone:
        try:
            adherence_rate = len([a for a in st.session_state.adherence if a['status'] == 'taken']) / len(st.session_state.adherence) * 100 if st.session_state.adherence else 0
            st.metric("Adherence Rate", f"{adherence_rate:.1f}%")
            next_reminder = min([datetime.fromisoformat(s) for s in st.session_state.schedule.values()]) if st.session_state.schedule else "None"
            st.metric("Next Reminder", next_reminder)
        except:
            st.error("Could not load dashboard data")

if __name__ == "__main__":
    # To run the backend: uvicorn app:app --reload
    # To run the frontend: streamlit run streamlit_app.py
    pass