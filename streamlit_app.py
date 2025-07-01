# streamlit_app.py
import os
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import websocket
import threading
import time
import threading
import pytz
import dotenv

# Load environment variables
dotenv.load_dotenv(verbose=True)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:3000")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:3000/ws/")
LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo

# Initialize session state
if "adherence_logs" not in st.session_state:
    st.session_state.adherence_logs = []
if "calls" not in st.session_state:
    st.session_state.calls = []
if "active_call_transcript" not in st.session_state:
    st.session_state.active_call_transcript = []
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "next_reminder" not in st.session_state:
    st.session_state.next_reminder = None
if "patient_info" not in st.session_state:
    st.session_state.patient_info = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None

# WebSocket connection handler
class WebSocketClient:
    def __init__(self, url, on_message=None, on_error=None, on_close=None):
        self.url = url
        self.ws = None
        self.on_message = on_message or self.default_on_message
        self.on_error = on_error or self.default_on_error
        self.on_close = on_close or self.default_on_close
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.run_forever)
        self.thread.daemon = True
        self.thread.start()

    def run_forever(self):
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                self.ws.run_forever()
                # If we get here, the connection was closed
                time.sleep(5)  # Wait before reconnecting
            except Exception as e:
                st.session_state.error_message = f"WebSocket error: {str(e)}"
                time.sleep(5)  # Wait before retrying

    def send(self, data):
        if self.ws:
            try:
                self.ws.send(json.dumps(data))
                return True
            except Exception as e:
                st.session_state.error_message = f"Error sending message: {str(e)}"
                return False
        return False

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

    def default_on_message(self, ws, message):
        pass

    def default_on_error(self, ws, error):
        st.session_state.error_message = f"WebSocket error: {str(error)}"

    def default_on_close(self, ws, close_status_code, close_msg):
        st.session_state.ws_connected = False

# WebSocket message handler
def on_message(ws, message):
    try:
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "schedule":
            next_reminder = data["data"].get("next_reminder")
            if next_reminder:
                st.session_state.next_reminder = datetime.fromisoformat(next_reminder)
            st.session_state.patient_info = data["data"]
            st.session_state.ws_connected = True
            
        elif message_type == "adherence_history":
            st.session_state.adherence_logs = data["data"]
            
        elif message_type == "reminder_sent":
            # Add to logs and update next reminder
            if "next_reminder" in data["data"]:
                st.session_state.next_reminder = datetime.fromisoformat(data["data"]["next_reminder"])
            
        elif message_type == "adherence_update":
            st.session_state.adherence_logs.append(data["data"])
            
        elif message_type == "call_update":
            # Update the transcript for an active call
            st.session_state.active_call_transcript = data["data"]["transcript"]
            
            # Also update calls list
            refresh_calls()
    except Exception as e:
        st.session_state.error_message = f"Error handling message: {str(e)}"

# Helper functions
def format_timestamp(timestamp_str):
    """Format ISO timestamp to readable local time"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        # Convert to local timezone
        return dt.astimezone(LOCAL_TIMEZONE).strftime("%Y-%m-%d %I:%M %p")
    except:
        return timestamp_str

def connect_websocket(phone_number):
    """Connect to WebSocket for real-time updates"""
    url = f"{WEBSOCKET_URL}{phone_number}"
    ws_client = WebSocketClient(
        url=url,
        on_message=on_message
    )
    ws_client.start()
    return ws_client

def schedule_medication():
    """Schedule medication reminders"""
    try:
        system_message = f"""You are a medication reminder assistant for {st.session_state.patient_name}. 
        Your job is to help them take their {st.session_state.medication} {st.session_state.dosage} on time. 
        Be friendly but direct. If they confirm taking their medication, log it as 'taken'.
        If they say they'll take it later, remind them of the importance of medication adherence.
        If they say they missed or skipped a dose, log it as 'missed' and advise them according to general medical guidelines.
        Always be respectful and supportive. Do not provide specific medical advice beyond general adherence guidance."""
        
        # Convert time inputs to proper format
        schedule = {}
        if st.session_state.morning_dose:
            schedule["morning"] = st.session_state.morning_time.strftime("%H:%M")
        if st.session_state.afternoon_dose:
            schedule["afternoon"] = st.session_state.afternoon_time.strftime("%H:%M")
        if st.session_state.evening_dose:
            schedule["evening"] = st.session_state.evening_time.strftime("%H:%M")
        if st.session_state.night_dose:
            schedule["night"] = st.session_state.night_time.strftime("%H:%M")
            
        if not schedule:
            st.error("Please select at least one dose time")
            return
            
        # Setup the request payload
        payload = {
            "patient": st.session_state.patient_name,
            "medication": st.session_state.medication,
            "dosage": st.session_state.dosage,
            "schedule": schedule,
            "phone": st.session_state.phone_number,
            "system_message": system_message
        }
        
        # Send the request
        response = requests.post(
            f"{API_URL}/schedule",
            json=payload
        )
        
        if response.status_code == 200:
            st.success("Medication reminders scheduled successfully!")
            st.session_state.ws_client = connect_websocket(st.session_state.phone_number)
            return True
        else:
            st.error(f"Error: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def log_adherence(status):
    """Log medication adherence"""
    try:
        payload = {
            "phone": st.session_state.phone_number,
            "status": status
        }
        
        response = requests.post(
            f"{API_URL}/log-adherence",
            json=payload
        )
        
        if response.status_code == 200:
            st.success(f"Medication marked as {status}")
            return True
        else:
            st.error(f"Error: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def initiate_call():
    """Initiate a call to the patient"""
    try:
        payload = {
            "phone": st.session_state.phone_number,
            "message": f"Hello {st.session_state.patient_name}, this is your medication reminder service calling about your {st.session_state.medication}."
        }
        
        response = requests.post(
            f"{API_URL}/call",
            json=payload
        )
        
        if response.status_code == 200:
            call_sid = response.json().get("call_sid")
            st.success(f"Call initiated with ID: {call_sid}")
            refresh_calls()
            return True
        else:
            st.error(f"Error: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def refresh_calls():
    """Refresh the list of active calls"""
    try:
        response = requests.get(f"{API_URL}/calls")
        if response.status_code == 200:
            st.session_state.calls = response.json()
        else:
            st.error(f"Error refreshing calls: {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def cancel_call(call_sid):
    """Cancel an active call"""
    try:
        response = requests.delete(f"{API_URL}/calls/{call_sid}")
        if response.status_code == 204:
            st.success("Call canceled successfully")
            refresh_calls()
            return True
        else:
            st.error(f"Error: {response.text}")
            return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def view_call_transcript(call_sid):
    """View transcript for a specific call"""
    try:
        response = requests.get(f"{API_URL}/calls/{call_sid}")
        if response.status_code == 200:
            call_data = response.json()
            st.session_state.active_call_transcript = call_data.get("transcript", [])
            return True
        else:
            st.error(f"Error: {response.text}")
            return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def get_adherence_logs():
    """Get adherence logs for the patient"""
    try:
        payload = {
            "phone": st.session_state.phone_number
        }
        response = requests.post(
            f"{API_URL}/adherence",
            json=payload
        )
        if response.status_code == 200:
            st.session_state.adherence_logs = response.json()
        else:
            st.error(f"Error retrieving adherence logs: {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Application Layout
st.title("Medication Reminder System")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Phone Number Input
    phone_input = st.text_input("Phone Number (E.164 format, e.g., +1234567890)", 
                               placeholder="+1234567890",
                               help="Must start with '+' and country code")
    
    # Configuration tab
    if phone_input and phone_input.startswith("+"):
        st.session_state.phone_number = phone_input
        
        if st.button("Connect"):
            st.session_state.ws_client = connect_websocket(st.session_state.phone_number)
            get_adherence_logs()
            refresh_calls()
    
    # Error message display
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        if st.button("Clear Error"):
            st.session_state.error_message = None

# Main area tabs
tab1, tab2, tab3, tab4 = st.tabs(["Schedule Medication", "Adherence", "Calls", "Dashboard"])

# Tab 1: Schedule Medication
with tab1:
    st.header("Schedule Medication Reminders")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.patient_name = st.text_input("Patient Name", key="patient_name_input")
        st.session_state.medication = st.text_input("Medication Name", key="medication_input")
    
    with col2:
        st.session_state.dosage = st.text_input("Dosage", key="dosage_input", placeholder="e.g., 10mg")
    
    st.subheader("Dosage Schedule")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.morning_dose = st.checkbox("Morning Dose", key="morning_checkbox")
        if st.session_state.morning_dose:
            st.session_state.morning_time = st.time_input("Morning Time", datetime.strptime("08:00", "%H:%M").time(), key="morning_time_input")
            
        st.session_state.afternoon_dose = st.checkbox("Afternoon Dose", key="afternoon_checkbox")
        if st.session_state.afternoon_dose:
            st.session_state.afternoon_time = st.time_input("Afternoon Time", datetime.strptime("13:00", "%H:%M").time(), key="afternoon_time_input")
    
    with col2:
        st.session_state.evening_dose = st.checkbox("Evening Dose", key="evening_checkbox")
        if st.session_state.evening_dose:
            st.session_state.evening_time = st.time_input("Evening Time", datetime.strptime("18:00", "%H:%M").time(), key="evening_time_input")
            
        st.session_state.night_dose = st.checkbox("Night Dose", key="night_checkbox")
        if st.session_state.night_dose:
            st.session_state.night_time = st.time_input("Night Time", datetime.strptime("22:00", "%H:%M").time(), key="night_time_input")
    
    if st.button("Schedule Reminders", use_container_width=True):
        if (st.session_state.patient_name and 
            st.session_state.medication and 
            st.session_state.dosage and 
            st.session_state.phone_number and
            (st.session_state.morning_dose or st.session_state.afternoon_dose or 
             st.session_state.evening_dose or st.session_state.night_dose)):
            schedule_medication()
        else:
            st.error("Please fill in all required fields and select at least one dose time")

# Tab 2: Adherence
with tab2:
    st.header("Medication Adherence")
    
    # Current medication info
    if st.session_state.patient_info:
        st.subheader("Current Medication")