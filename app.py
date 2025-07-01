# app.py
import os
import logging
import asyncio
import base64
import json
from collections import deque
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import urllib3
import dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.base.exceptions import TwilioRestException

from services.llm_service import LLMFactory
from services.call_context import CallContext

# Disable SSL warnings for local development (ngrok uses self-signed certificates)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

dotenv.load_dotenv(verbose=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("App")

# Create a BackgroundScheduler instance
scheduler = BackgroundScheduler()

# Models
class MedicationRequest(BaseModel):
    patient: str = Field(..., description="Patient's name")
    medication: str = Field(..., description="Medication name")
    dosage: str = Field(..., description="Medication dosage")
    schedule: Dict[str, str] = Field(..., description="Medication schedule with time entries in HH:MM format")
    phone: str = Field(..., description="Patient's phone number in E.164 format (e.g., +1234567890)")
    system_message: str = Field(..., description="System prompt for the LLM assistant")
    
    @validator('phone')
    def validate_phone(cls, v):
        if not v.startswith('+'):
            raise ValueError('Phone number must be in E.164 format starting with + sign')
        return v
    
    @validator('schedule')
    def validate_schedule(cls, v):
        for time_str in v.values():
            try:
                datetime.strptime(time_str, "%H:%M")
            except ValueError:
                raise ValueError(f"Invalid time format: {time_str}. Use HH:MM format.")
        return v

class CallRequest(BaseModel):
    phone: str = Field(..., description="Phone number to call in E.164 format")
    message: str = Field(..., description="Initial message for the call")
    
    @validator('phone')
    def validate_phone(cls, v):
        if not v.startswith('+'):
            raise ValueError('Phone number must be in E.164 format starting with + sign')
        return v

class AdherenceLog(BaseModel):
    phone: str = Field(..., description="Patient's phone number")
    start_date: Optional[datetime] = Field(None, description="Start date for filtering logs")
    end_date: Optional[datetime] = Field(None, description="End date for filtering logs")

# State Management
active_calls: Dict[str, Dict] = {}
medication_schedules: Dict[str, Dict] = {}
adherence_logs: Dict[str, List] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# Initialize Twilio client
try:
    twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {str(e)}")
    twilio_client = None

# Lifespan event handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    if not scheduler.running:
        try:
            scheduler.start()
            logger.info("Scheduler started successfully")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
    yield
    # On shutdown
    if scheduler.running:
        try:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {str(e)}")

app = FastAPI(
    title="Medication Reminder API",
    description="API for managing medication reminders and adherence",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "scheduler_running": scheduler.running}

@app.post("/schedule", response_model=Dict)
async def schedule_medication(request: MedicationRequest, background_tasks: BackgroundTasks):
    """Schedule medication reminders for a patient"""
    try:
        # Create CallContext object
        context = CallContext(
            system_message=request.system_message,
            initial_message=f"Hello {request.patient}, this is your medication reminder service.",
            medication=request.medication,
            dosage=request.dosage,
            schedule=request.schedule
        )
        
        # Store the context in medication_schedules
        medication_schedules[request.phone] = {
            "context": context,
            "patient": request.patient,
            "last_reminder": None,
            "next_reminder": calculate_next_reminder(request.schedule)
        }
        
        # Initialize adherence log if not exists
        if request.phone not in adherence_logs:
            adherence_logs[request.phone] = []
        
        # Clear existing jobs for this phone number
        background_tasks.add_task(clear_existing_jobs, request.phone)
        
        # Schedule new jobs
        for dose, time_str in request.schedule.items():
            hour, minute = map(int, time_str.split(":"))
            job_id = f"{request.phone}_{dose}"
            
            try:
                scheduler.add_job(
                    send_reminder,
                    'cron',
                    hour=hour,
                    minute=minute,
                    args=[request.phone, dose],
                    id=job_id,
                    replace_existing=True
                )
                logger.info(f"Scheduled reminder for {request.patient}: {dose} dose at {time_str}")
            except Exception as e:
                logger.error(f"Failed to schedule job {job_id}: {str(e)}")
        
        return {
            "status": "scheduled",
            "patient": request.patient,
            "medication": request.medication,
            "next_reminder": medication_schedules[request.phone]["next_reminder"].isoformat() if medication_schedules[request.phone]["next_reminder"] else None
        }
        
    except Exception as e:
        logger.error(f"Scheduling failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to schedule medication: {str(e)}")

def clear_existing_jobs(phone: str):
    """Clear existing jobs for a phone number"""
    try:
        for job in scheduler.get_jobs():
            if job.id.startswith(f"{phone}_"):
                scheduler.remove_job(job.id)
                logger.info(f"Removed existing job: {job.id}")
    except Exception as e:
        logger.error(f"Error clearing jobs for {phone}: {str(e)}")

@app.post("/call", response_model=Dict)
async def initiate_call(request: CallRequest):
    """Initiate a call to a patient"""
    if not twilio_client:
        raise HTTPException(status_code=503, detail="Twilio client not available")
        
    try:
        if not all([os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"), os.getenv("TWILIO_PHONE_NUMBER")]):
            raise HTTPException(status_code=503, detail="Twilio credentials not configured")
            
        twiml = generate_twiml(request.message)
        callback_url = os.getenv("CALLBACK_URL", "http://localhost:3000") + "/twilio-webhook"
        
        try:
            call = twilio_client.calls.create(
                twiml=twiml,
                to=request.phone,
                from_=os.getenv("TWILIO_PHONE_NUMBER"),
                status_callback=callback_url,
                status_callback_event=["initiated", "ringing", "answered", "completed"]
            )
            
            # Create call context
            context = None
            if request.phone in medication_schedules:
                context = medication_schedules[request.phone]["context"]
            
            # Store call information
            active_calls[call.sid] = {
                "phone": request.phone,
                "transcript": [],
                "status": "initiated",
                "created_at": datetime.now().isoformat(),
                "context": context
            }
            
            logger.info(f"Call initiated: {call.sid} to {request.phone}")
            return {"call_sid": call.sid, "status": "initiated"}
            
        except TwilioRestException as te:
            logger.error(f"Twilio error: {str(te)}")
            raise HTTPException(status_code=400, detail=f"Twilio error: {str(te)}")
            
    except Exception as e:
        logger.error(f"Call failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Call failed: {str(e)}")

@app.get("/calls", response_model=List[Dict])
async def list_calls():
    """List all active calls"""
    return [
        {
            "call_sid": call_sid,
            "phone": call_data["phone"],
            "status": call_data["status"],
            "created_at": call_data["created_at"]
        }
        for call_sid, call_data in active_calls.items()
    ]

@app.get("/calls/{call_sid}", response_model=Dict)
async def get_call(call_sid: str):
    """Get details of a specific call"""
    if call_sid not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
        
    return active_calls[call_sid]

@app.delete("/calls/{call_sid}", status_code=204)
async def delete_call(call_sid: str):
    """Delete a call from active calls"""
    if call_sid not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
        
    try:
        call = twilio_client.calls(call_sid).fetch()
        if call.status not in ["completed", "failed", "canceled"]:
            call.update(status="completed")
    except Exception as e:
        logger.error(f"Error terminating call {call_sid}: {str(e)}")
        
    active_calls.pop(call_sid)
    return Response(status_code=204)

@app.post("/adherence", response_model=List[Dict])
async def get_adherence_logs(log_request: AdherenceLog):
    """Get adherence logs for a patient"""
    if log_request.phone not in adherence_logs:
        return []
        
    logs = adherence_logs[log_request.phone]
    
    # Filter by date range if provided
    if log_request.start_date or log_request.end_date:
        filtered_logs = []
        for log in logs:
            log_time = datetime.fromisoformat(log["timestamp"])
            if log_request.start_date and log_time < log_request.start_date:
                continue
            if log_request.end_date and log_time > log_request.end_date:
                continue
            filtered_logs.append(log)
        return filtered_logs
        
    return logs

@app.post("/log-adherence", response_model=Dict)
async def log_adherence(data: Dict):
    """Log medication adherence"""
    phone = data.get("phone")
    status = data.get("status")
    
    if not phone or not status:
        raise HTTPException(status_code=400, detail="Phone and status are required")
        
    if phone not in medication_schedules:
        raise HTTPException(status_code=404, detail="No medication schedule found for this phone number")
        
    context = medication_schedules[phone]["context"]
    dose_time = datetime.now()
    
    # Create log entry
    log_entry = {
        "timestamp": dose_time.isoformat(),
        "medication": context.medication,
        "dose": context.dosage,
        "status": status
    }
    
    # Add to logs
    if phone not in adherence_logs:
        adherence_logs[phone] = []
    adherence_logs[phone].append(log_entry)
    
    # Update schedule with last logged time
    medication_schedules[phone]["last_reminder"] = dose_time
    medication_schedules[phone]["next_reminder"] = calculate_next_reminder(context.schedule)
    
    # Broadcast to connected websockets
    if phone in websocket_connections:
        for websocket in websocket_connections[phone]:
            try:
                await websocket.send_json({
                    "type": "adherence_update",
                    "data": log_entry
                })
            except Exception as e:
                logger.error(f"Error sending to websocket: {str(e)}")
    
    return log_entry

@app.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    # Add websocket to connections
    if phone not in websocket_connections:
        websocket_connections[phone] = []
    websocket_connections[phone].append(websocket)
    
    try:
        # Send initial data
        if phone in medication_schedules:
            await websocket.send_json({
                "type": "schedule",
                "data": {
                    "next_reminder": medication_schedules[phone]["next_reminder"].isoformat() if medication_schedules[phone]["next_reminder"] else None,
                    "patient": medication_schedules[phone].get("patient", ""),
                    "medication": medication_schedules[phone]["context"].medication,
                    "dosage": medication_schedules[phone]["context"].dosage,
                    "schedule": medication_schedules[phone]["context"].schedule
                }
            })
        
        if phone in adherence_logs:
            await websocket.send_json({
                "type": "adherence_history",
                "data": adherence_logs[phone]
            })
        
        # Handle messages
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            # Handle different message types
            if data_json.get("type") == "log_adherence":
                log_entry = {
                    "phone": phone,
                    "status": data_json.get("status", "unknown")
                }
                response = await log_adherence(log_entry)
                await websocket.send_json({"type": "adherence_logged", "data": response})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {phone}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Remove websocket from connections
        if phone in websocket_connections and websocket in websocket_connections[phone]:
            websocket_connections[phone].remove(websocket)

@app.post("/twilio-webhook")
async def twilio_webhook(request_data: dict):
    """Handle Twilio webhook callbacks"""
    call_sid = request_data.get("CallSid")
    speech_result = request_data.get("SpeechResult")
    call_status = request_data.get("CallStatus")
    
    # Log the webhook data
    logger.info(f"Twilio webhook: {call_sid} - Status: {call_status} - Speech: {speech_result}")
    
    if not call_sid:
        return str(VoiceResponse())
    
    # Update call status
    if call_sid in active_calls:
        active_calls[call_sid]["status"] = call_status
    
    # Handle speech recognition result
    if speech_result and call_sid in active_calls:
        try:
            # Get the context for this call
            context = active_calls[call_sid].get("context")
            if not context:
                phone = active_calls[call_sid].get("phone")
                if phone in medication_schedules:
                    context = medication_schedules[phone]["context"]
            
            # Create or get LLM service
            provider = os.getenv("LLM_SERVICE", "openai")
            llm_service = LLMFactory.get_service(provider, context) if context else None
            
            # Record user speech in transcript
            active_calls[call_sid]["transcript"].append({
                "role": "user",
                "content": speech_result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate LLM response
            response_text = "I'm sorry, I couldn't process your request."
            if llm_service:
                interaction_count = len(active_calls[call_sid]["transcript"]) // 2
                response_text = await llm_service.completion(speech_result, interaction_count)
            
            # Record assistant response in transcript
            active_calls[call_sid]["transcript"].append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send response
            vr = VoiceResponse()
            gather = Gather(input="speech", action="/twilio-webhook", method="POST", timeout=5, speech_timeout="auto")
            gather.say(response_text)
            vr.append(gather)
            
            # Broadcast to websockets
            phone = active_calls[call_sid].get("phone")
            if phone in websocket_connections:
                for websocket in websocket_connections[phone]:
                    try:
                        await websocket.send_json({
                            "type": "call_update",
                            "data": {
                                "call_sid": call_sid,
                                "transcript": active_calls[call_sid]["transcript"][-2:]
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error sending to websocket: {str(e)}")
            
            return str(vr)
            
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
    
    # Default response
    vr = VoiceResponse()
    gather = Gather(input="speech", action="/twilio-webhook", method="POST")
    gather.say("I'm listening. Please speak after the tone.")
    vr.append(gather)
    return str(vr)

def generate_twiml(message: str):
    """Generate TwiML for a call"""
    vr = VoiceResponse()
    gather = Gather(input="speech", action="/twilio-webhook", method="POST", timeout=5, speech_timeout="auto")
    gather.say(message)
    vr.append(gather)
    # Add fallback for no input
    vr.say("I didn't hear anything. Goodbye.")
    vr.hangup()
    return str(vr)

def calculate_next_reminder(schedule: Dict[str, str]) -> Optional[datetime]:
    """Calculate the next scheduled reminder time based on the current time"""
    now = datetime.now()
    times = []
    
    for time_str in schedule.values():
        try:
            time_obj = datetime.strptime(time_str, "%H:%M").time()
            date_time = datetime.combine(now.date(), time_obj)
            if date_time < now:
                # If time already passed today, schedule for tomorrow
                date_time = datetime.combine(now.date() + timedelta(days=1), time_obj)
            times.append(date_time)
        except ValueError as e:
            logger.error(f"Invalid time format: {time_str} - {str(e)}")
    
    if not times:
        return None
        
    return min(times)

async def send_reminder(phone: str, dose: str):
    """Send a reminder for a medication dose"""
    try:
        if phone not in medication_schedules:
            logger.error(f"No medication schedule found for {phone}")
            return
            
        context = medication_schedules[phone]["context"]
        patient_name = medication_schedules[phone].get("patient", "")
        
        # Create message
        reminder_message = f"Hello {patient_name}, this is your reminder for the {dose} dose of {context.medication} ({context.dosage})."
        
        # Log the reminder
        logger.info(f"Sending reminder to {phone}: {reminder_message}")
        
        # Update schedule info
        now = datetime.now()
        medication_schedules[phone]["last_reminder"] = now
        medication_schedules[phone]["next_reminder"] = calculate_next_reminder(context.schedule)
        
        # Make call
        try:
            if twilio_client:
                call = twilio_client.calls.create(
                    twiml=generate_twiml(reminder_message),
                    to=phone,
                    from_=os.getenv("TWILIO_PHONE_NUMBER"),
                    status_callback=os.getenv("CALLBACK_URL", "http://localhost:3000") + "/twilio-webhook",
                    status_callback_event=["initiated", "ringing", "answered", "completed"]
                )
                
                # Store call information
                active_calls[call.sid] = {
                    "phone": phone,
                    "transcript": [],
                    "status": "initiated",
                    "created_at": now.isoformat(),
                    "context": context,
                    "reminder_for": dose
                }
                
                logger.info(f"Reminder call initiated: {call.sid} to {phone}")
            else:
                logger.error("Twilio client not available for reminder call")
        except Exception as e:
            logger.error(f"Failed to make reminder call: {str(e)}")
        
        # Notify connected websockets
        if phone in websocket_connections:
            notification = {
                "type": "reminder_sent",
                "data": {
                    "timestamp": now.isoformat(),
                    "medication": context.medication,
                    "dose": dose,
                    "dosage": context.dosage,
                    "next_reminder": medication_schedules[phone]["next_reminder"].isoformat() if medication_schedules[phone]["next_reminder"] else None
                }
            }
            
            for websocket in websocket_connections[phone]:
                try:
                    await websocket.send_json(notification)
                except Exception as e:
                    logger.error(f"Error sending to websocket: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in send_reminder: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)