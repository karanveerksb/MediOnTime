first i gave you this prompt "this is code of services/llm_sevice.py "import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import anthropic
from openai import AsyncOpenAI
import google.generativeai as genai
from google.generativeai.types import ContentDict

from functions.function_manifest import tools
from logger_config import get_logger
from services.call_context import CallContext
from services.event_emmiter import EventEmitter

logger = get_logger("LLMService")

class AbstractLLMService(EventEmitter, ABC):
    def __init__(self, context: CallContext):
        super().__init__()
        self.context = context
        self.system_message = context.system_message
        self.initial_message = context.initial_message
        self.user_context = []
        self.partial_response_index = 0
        self.sentence_buffer = ""
        self.available_functions = self._load_functions()
        self.medication_schedule = context.schedule
        self.adherence_log = []

    def _load_functions(self):
        functions = {}
        for tool in tools:
            module = importlib.import_module(f'functions.{tool["function"]["name"]}')
            functions[tool["function"]["name"]] = getattr(module, tool["function"]["name"])
        return functions

    def update_context(self, context: CallContext):
        self.context = context
        self.system_message = context.system_message
        self.initial_message = context.initial_message
        self.medication_schedule = context.schedule

    def log_adherence(self, status: str, dose_time: datetime):
        self.adherence_log.append({
            "timestamp": dose_time.isoformat(),
            "medication": self.context.medication,
            "dose": self.context.dosage,
            "status": status
        })

    def calculate_next_dose(self, current_time: datetime) -> Optional[datetime]:
        schedule = self.medication_schedule
        times = [datetime.strptime(t, "%H:%M").time() for t in schedule.values()]
        current_time = current_time.time()
        
        for t in sorted(times):
            if current_time < t:
                return datetime.combine(datetime.today(), t)
        return datetime.combine(datetime.today() + timedelta(days=1), min(times))

    async def handle_medication_reminder(self, interaction_count: int):
        next_dose = self.calculate_next_dose(datetime.now())
        if next_dose:
            reminder = f"Time for your {self.context.medication} ({self.context.dosage}). Please confirm you've taken it."
            await self.completion(reminder, interaction_count, role="system")

    @abstractmethod
    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        pass

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s]

    async def emit_complete_sentences(self, text: str, interaction_count: int):
        self.sentence_buffer += text
        sentences = self.split_into_sentences(self.sentence_buffer)
        
        for sentence in sentences[:-1]:
            await self.emit('llmreply', {
                "partialResponseIndex": self.partial_response_index,
                "partialResponse": sentence
            }, interaction_count)
            self.partial_response_index += 1
        
        self.sentence_buffer = sentences[-1] if sentences else ""

class OpenAIService(AbstractLLMService):
    def __init__(self, context: CallContext):
        super().__init__(context)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        try:
            self.user_context.append({"role": role, "content": text, "name": name})
            messages = [{"role": "system", "content": self.system_message}] + self.user_context

            stream = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True
            )

            complete_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    complete_response += content
                    await self.emit_complete_sentences(content, interaction_count)
                
                if chunk.choices[0].delta.tool_calls:
                    await self.handle_tool_calls(chunk.choices[0].delta.tool_calls, interaction_count)

            if complete_response.strip():
                self.user_context.append({"role": "assistant", "content": complete_response})

        except Exception as e:
            logger.error(f"OpenAI Error: {str(e)}")

    async def handle_tool_calls(self, tool_calls, interaction_count):
        for tool in tool_calls:
            function_name = tool.function.name
            function_args = json.loads(tool.function.arguments)
            
            if function_name in self.available_functions:
                response = await self.available_functions[function_name](self.context, **function_args)
                await self.emit('llmreply', {
                    "partialResponseIndex": None,
                    "partialResponse": response.get('confirmation', '')
                }, interaction_count)

class AnthropicService(AbstractLLMService):
    def __init__(self, context: CallContext):
        super().__init__(context)
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        try:
            self.user_context.append({"role": role, "content": text})
            
            async with self.client.messages.stream(
                model="claude-3-opus-20240229",
                max_tokens=400,
                system=self.system_message,
                messages=self.user_context,
                tools=self._convert_tools(),
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        await self.emit_complete_sentences(event.delta.text, interaction_count)
                    elif event.type == "tool_use":
                        await self.handle_tool_use(event, interaction_count)

        except Exception as e:
            logger.error(f"Anthropic Error: {str(e)}")

    def _convert_tools(self):
        return [{
            "name": tool['function']['name'],
            "description": tool['function']['description'],
            "input_schema": tool['function']['parameters']
        } for tool in tools]

    async def handle_tool_use(self, event, interaction_count):
        function_name = event.tool_name
        function_args = event.input
        
        if function_name in self.available_functions:
            response = await self.available_functions[function_name](self.context, **function_args)
            await self.emit('llmreply', {
                "partialResponseIndex": None,
                "partialResponse": response.get('confirmation', '')
            }, interaction_count)

class GeminiService(AbstractLLMService):
    def __init__(self, context: CallContext):
        super().__init__(context)
        self._configure_gemini()
        self.model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            tools=self._convert_tools()
        )
        self.conversation = self.model.start_chat(history=[])

    def _configure_gemini(self):
        """Initialize Gemini with proper error handling"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing in environment variables")
        genai.configure(api_key=api_key)

    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        try:
            # Build conversation history
            self._update_conversation_history(role, text)
            
            # Generate response with tool handling
            async for response in await self.conversation.send_message_async(text, stream=True):
                await self._process_response(response, interaction_count)

        except Exception as e:
            logger.error(f"Gemini Error: {str(e)}")
            await self.emit('llmreply', {
                "partialResponseIndex": None,
                "partialResponse": "Sorry, I'm having trouble processing your request."
            }, interaction_count)

    def _update_conversation_history(self, role: str, text: str):
        """Maintain conversation context"""
        self.user_context.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": text}]
        })

    async def _process_response(self, response, interaction_count):
        """Handle different response types from Gemini"""
        for part in response.candidates[0].content.parts:
            if part.text:
                await self._handle_text_response(part.text, interaction_count)
            elif part.function_call:
                await self._handle_function_call(part.function_call, interaction_count)

    async def _handle_text_response(self, text: str, interaction_count: int):
        """Process text responses with streaming"""
        await self.emit_complete_sentences(text, interaction_count)
        self.user_context.append({
            "role": "model",
            "parts": [{"text": text}]
        })

    async def _handle_function_call(self, function_call, interaction_count: int):
        """Execute and respond to function calls"""
        try:
            function_name = function_call.name
            function_args = json.loads(function_call.args)
            
            if function_name in self.available_functions:
                result = await self.available_functions[function_name](self.context, **function_args)
                
                # Send confirmation to client
                await self.emit('llmreply', {
                    "partialResponseIndex": None,
                    "partialResponse": result.get('confirmation', 'Action completed')
                }, interaction_count)
                
                # Add function result to conversation
                self.conversation.send_message(json.dumps(result))

        except Exception as e:
            logger.error(f"Function call failed: {str(e)}")
            await self.emit('llmreply', {
                "partialResponseIndex": None,
                "partialResponse": "Failed to execute action"
            }, interaction_count)

    def _convert_tools(self):
        """Convert OpenAI-style tools to Gemini format"""
        return [genai.Tool(
            function_declarations=[genai.FunctionDeclaration(
                name=tool['function']['name'],
                description=tool['function']['description'],
                parameters=self._adapt_parameters(tool['function']['parameters'])
            )]
        ) for tool in tools]

    def _adapt_parameters(self, params: dict) -> dict:
        """Convert parameters to Gemini-compatible format"""
        return {
            "type_": "object",
            "properties": {
                prop: {"type_": details["type"]} 
                for prop, details in params.get("properties", {}).items()
            },
            "required": params.get("required", [])
        }

class LLMFactory:
    @staticmethod
    def get_service(provider: str, context: CallContext) -> AbstractLLMService:
        provider = provider.lower()
        if provider == "openai":
            return OpenAIService(context)
        elif provider == "anthropic":
            return AnthropicService(context)
        elif provider == "gemini":
            return GeminiService(context)
        raise ValueError(f"Unsupported provider: {provider}")"  this is code for app.py "import os
import logging
import asyncio
import base64
import json
from collections import deque
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import urllib3
import dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather

from services.llm_service import LLMFactory
from services.call_context import CallContext

# Disable SSL warnings for local development (ngrok uses self-signed certificates)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

dotenv.load_dotenv(verbose=True)

# Set up logging
logger = logging.getLogger("App")
logger.setLevel(logging.INFO)

# Create a BackgroundScheduler instance
scheduler = BackgroundScheduler()

# Lifespan event handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started")
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class MedicationRequest(BaseModel):
    patient: str
    medication: str
    dosage: str
    schedule: dict
    phone: str
    system_message: str

class CallRequest(BaseModel):
    phone: str
    message: str

# State Management
active_calls = {}
medication_schedules = {}

twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

@app.post("/schedule")
async def schedule_medication(request: MedicationRequest):
    try:
        # Validate schedule format
        for dose, time_str in request.schedule.items():
            if not validate_time_format(time_str):
                raise ValueError(f"Invalid time format for {dose}: {time_str}. Use HH:MM")
        # Create a CallContext for scheduling reminders
        context = CallContext(
            system_message=request.system_message,
            medication=request.medication,
            dosage=request.dosage,
            schedule=request.schedule
        )
        medication_schedules[request.phone] = {
            "context": context,
            "last_reminder": None,
            "next_reminder": calculate_next_reminder(request.schedule)
        }
        # Clear any existing jobs for this phone number
        scheduler.remove_all_jobs()
        # Add new jobs for each scheduled dose
        for dose, time_str in request.schedule.items():
            hour, minute = map(int, time_str.split(":"))
            scheduler.add_job(
                send_reminder,
                'cron',
                hour=hour,
                minute=minute,
                args=[request.phone, dose],
                id=f"{request.phone}_{dose}"
            )
        return {"status": "scheduled", "next_reminder": medication_schedules[request.phone]["next_reminder"]}
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(400, detail=str(ve))
    except Exception as e:
        logger.error(f"Scheduling failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail="Failed to schedule medication. Check server logs.")

def validate_time_format(time_str: str) -> bool:
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False

@app.post("/call")
async def initiate_call(request: CallRequest):
    try:
        if not request.phone.startswith("+"):
            raise ValueError("Phone number must be in E.164 format (e.g., +1234567890)")
        if not all([os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"), os.getenv("TWILIO_PHONE_NUMBER")]):
            raise EnvironmentError("Twilio credentials not configured")
        call = twilio_client.calls.create(
            twiml=generate_twiml(request.message),
            to=request.phone,
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            status_callback="http://localhost:3000/twilio-webhook",
            status_callback_event=["initiated", "ringing", "answered", "completed"]
        )
        active_calls[call.sid] = {
            "transcript": [],
            "status": "initiated",
            "created_at": datetime.now().isoformat()
        }
        return {"call_sid": call.sid, "status": "initiated"}
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(400, detail=str(ve))
    except Exception as e:
        logger.error(f"Call failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Call failed: {str(e)}")

@app.websocket("/ws/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if call_id in active_calls:
                active_calls[call_id]["transcript"].append({
                    "role": "user",
                    "content": data,
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {call_id}")

@app.post("/twilio-webhook")
async def handle_twilio_webhook(request: dict):
    call_sid = request.get("CallSid")
    speech = request.get("SpeechResult")
    if call_sid and speech:
        llm_service = LLMFactory.create(os.getenv("LLM_SERVICE", "openai"))
        response = await llm_service.generate_response(
            speech,
            active_calls[call_sid].get("context")
        )
        active_calls[call_sid]["transcript"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        vr = VoiceResponse()
        vr.say(response)
        return str(vr)
    return str(VoiceResponse())

def generate_twiml(message: str):
    vr = VoiceResponse()
    gather = Gather(input="speech", action="/twilio-webhook", method="POST")
    gather.say(message)
    vr.append(gather)
    return str(vr)

def calculate_next_reminder(schedule: dict):
    now = datetime.now()
    times = [datetime.strptime(t, "%H:%M").time() for t in schedule.values()]
    for t in sorted(times):
        if now.time() < t:
            return datetime.combine(now.date(), t)
    return datetime.combine(now.date() + timedelta(days=1), min(times))

async def send_reminder(phone: str, dose: str):
    if phone in medication_schedules:
        context = medication_schedules[phone]["context"]
        llm_service = LLMFactory.create(os.getenv("LLM_SERVICE", "openai"))
        reminder = f"Time for your {dose} dose of {context.medication}"
        await llm_service.send_reminder(reminder, context)
        medication_schedules[phone]["last_reminder"] = datetime.now()
        medication_schedules[phone]["next_reminder"] = calculate_next_reminder(context.schedule)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
"    this is code for streamlit_app.py " import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Get LLM Service from .env
LLM_SERVICE = os.getenv("LLM_SERVICE", "openai")

# Streamlit UI Config
st.set_page_config(page_title="Medication Manager", layout="wide")

# Function to fetch response from LLM Service
def fetch_llm_response(prompt):
    """Fetch response from selected LLM service."""
    api_url = os.getenv(f"{LLM_SERVICE.upper()}_API_URL")
    api_key = os.getenv(f"{LLM_SERVICE.upper()}_API_KEY")

    if not api_url or not api_key:
        st.error(f"Missing API configuration for {LLM_SERVICE}")
        return None

    try:
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"prompt": prompt}
        )
        if response.status_code == 200:
            return response.json().get("response", "No response received.")
        else:
            st.error(f"Error from {LLM_SERVICE}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None

# Main App
def main():
    st.title("💊 Smart Medication Reminder")
    
    with st.sidebar:
        st.header("Configuration")
        patient = st.text_input("Patient Name", "John Doe")
        medication = st.text_input("Medication", "Metformin 500mg")
        dosage = st.text_input("Dosage", "1 tablet")
        phone = st.text_input("Patient Phone", "+917986000941")
        
        st.subheader("Dosage Schedule")
        schedule = {
            "morning": st.time_input("Morning", value=datetime.strptime("08:00", "%H:%M")).strftime("%H:%M"),
            "afternoon": st.time_input("Afternoon", value=datetime.strptime("14:00", "%H:%M")).strftime("%H:%M"),
            "evening": st.time_input("Evening", value=datetime.strptime("20:00", "%H:%M")).strftime("%H:%M")
        }
        
        system_prompt = f"""You are an AI medication assistant for {patient}.
        Medication: {medication} ({dosage})
        Schedule: {', '.join([f"{k.capitalize()} {v}" for k, v in schedule.items()])}"""
        
        if st.button("Activate Schedule"):
            try:
                response = requests.post(
                    "http://localhost:3000/schedule",
                    json={
                        "patient": patient,
                        "medication": medication,
                        "dosage": dosage,
                        "schedule": schedule,
                        "phone": phone,
                        "system_message": system_prompt
                    }
                )
                if response.status_code == 200:
                    st.success("Schedule activated! 🎉")
                else:
                    st.error(f"Failed to activate schedule: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")

    # Layout
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        st.subheader("Active Reminders")
        for dose, time in schedule.items():
            st.info(f"{dose.capitalize()}: {time}")
                
        st.subheader("Adherence Tracking")
        st.metric("Current Streak", "5 days")
        st.progress(0.85)
    
    with main_col2:
        st.subheader("AI Medication Advice")
        advice = fetch_llm_response(f"What are the side effects of {medication}?")
        if advice:
            st.info(advice)

        st.subheader("Call Management")
        if st.button("Initiate Check-in Call"):
            try:
                response = requests.post(
                    "http://localhost:3000/call",
                    json={"phone": phone, "message": "Hello, time for your medication check-in"}
                )
                if response.status_code == 200:
                    st.success("Call initiated!")
                else:
                    st.error(f"Failed to initiate call: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")
        
        st.subheader("Recent Interactions")
        st.table([{
            "Time": "10:00 AM",
            "Message": "Taken medication with water"
        }])

if __name__ == "__main__":
    main()
" i am not able to understand why this streamlit_app requesting api key api url from user when there are already app.py and llm_service.py so therefore give me correct code for streamlit_app.py " 
then these are your replies "I can see that the problem with the streamlit_app.py is that it's trying to directly call an LLM service API rather than using the existing app.py and llm_service.py implementation. Let me provide a corrected version of the streamlit_app.py that properly integrates with your existing backend.

```python
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Streamlit UI Config
st.set_page_config(page_title="Medication Manager", layout="wide")

# Function to call the backend API
def call_backend_api(endpoint, data):
    """Call the backend API endpoint with provided data."""
    api_url = os.getenv("BACKEND_URL", "http://localhost:3000")
    
    try:
        response = requests.post(
            f"{api_url}/{endpoint}",
            json=data
        )
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None

# Main App
def main():
    st.title("💊 Smart Medication Reminder")
    
    with st.sidebar:
        st.header("Configuration")
        patient = st.text_input("Patient Name", "John Doe")
        medication = st.text_input("Medication", "Metformin 500mg")
        dosage = st.text_input("Dosage", "1 tablet")
        phone = st.text_input("Patient Phone", "+1234567890")
        
        st.subheader("Dosage Schedule")
        schedule = {
            "morning": st.time_input("Morning", value=datetime.strptime("08:00", "%H:%M")).strftime("%H:%M"),
            "afternoon": st.time_input("Afternoon", value=datetime.strptime("14:00", "%H:%M")).strftime("%H:%M"),
            "evening": st.time_input("Evening", value=datetime.strptime("20:00", "%H:%M")).strftime("%H:%M")
        }
        
        system_prompt = f"""You are an AI medication assistant for {patient}.
        Medication: {medication} ({dosage})
        Schedule: {', '.join([f"{k.capitalize()} {v}" for k, v in schedule.items()])}"""
        
        if st.button("Activate Schedule"):
            response = call_backend_api("schedule", {
                "patient": patient,
                "medication": medication,
                "dosage": dosage,
                "schedule": schedule,
                "phone": phone,
                "system_message": system_prompt
            })
            
            if response and response.status_code == 200:
                st.success("Schedule activated! 🎉")
                st.json(response.json())
            elif response:
                st.error(f"Failed to activate schedule: {response.text}")

    # Layout
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        st.subheader("Active Reminders")
        for dose, time in schedule.items():
            st.info(f"{dose.capitalize()}: {time}")
                
        st.subheader("Adherence Tracking")
        adherence_placeholder = st.empty()
        
        # Placeholder for adherence data - in a real app, you would fetch this from the backend
        adherence_placeholder.metric("Current Streak", "5 days")
        st.progress(0.85)
    
    with main_col2:
        st.subheader("AI Medication Advice")
        
        if st.button("Get Medication Information"):
            # This would ideally call a dedicated endpoint in your backend that uses LLMService
            call_data = {
                "phone": phone,
                "message": f"What are the important things to know about {medication}?"
            }
            response = call_backend_api("call", call_data)
            
            if response and response.status_code == 200:
                st.success("Information request sent!")
                st.json(response.json())
            elif response:
                st.error(f"Failed to get information: {response.text}")

        st.subheader("Call Management")
        if st.button("Initiate Check-in Call"):
            call_data = {
                "phone": phone,
                "message": f"Hello {patient}, this is your medication reminder for {medication}. Have you taken your {dosage} dose?"
            }
            response = call_backend_api("call", call_data)
            
            if response and response.status_code == 200:
                st.success("Call initiated!")
                st.json(response.json())
            elif response:
                st.error(f"Failed to initiate call: {response.text}")
        
        st.subheader("Recent Interactions")
        # In a complete implementation, you would fetch this from a backend endpoint
        st.table([{
            "Time": "10:00 AM",
            "Message": "Taken medication with water"
        }])

if __name__ == "__main__":
    main()

```

The key issues I fixed in the streamlit_app.py:

1. **Integration with your backend:** The original code was trying to directly call LLM APIs instead of using your existing backend services. The corrected version makes calls to your FastAPI backend endpoints.

2. **Removed direct LLM service calls:** I removed the `fetch_llm_response()` function that was trying to directly call external LLM APIs. Instead, your app.py already handles the LLM integration via the LLMFactory.

3. **Simplified configuration:** Instead of requiring API keys in the Streamlit app, it now simply needs the backend URL (defaulting to localhost:3000).

4. **Better error handling:** Added proper error handling for API calls.

5. **Improved UX flow:** Made the medication information request a button-triggered action rather than an automatic API call.

To use this, you'll just need to make sure your backend (app.py) is running, and set the BACKEND_URL environment variable if it's not running on the default http://localhost:3000.

Would you like me to explain any other aspects of the integration or make additional improvements to the code?"
 then i asked a question "can you provide all the three codes improved to their best possible version"
these are your replies "I'll provide improved versions of all three files: llm_service.py, app.py, and streamlit_app.py, addressing potential issues and enhancing functionality.

```python
import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import anthropic
from openai import AsyncOpenAI
import google.generativeai as genai
from google.generativeai.types import ContentDict

from functions.function_manifest import tools
from logger_config import get_logger
from services.call_context import CallContext
from services.event_emmiter import EventEmitter

logger = get_logger("LLMService")

class AbstractLLMService(EventEmitter, ABC):
    def __init__(self, context: CallContext):
        super().__init__()
        self.context = context
        self.system_message = context.system_message
        self.initial_message = context.initial_message
        self.user_context = []
        self.partial_response_index = 0
        self.sentence_buffer = ""
        self.available_functions = self._load_functions()
        self.medication_schedule = context.schedule
        self.adherence_log = []

    def _load_functions(self):
        """Load function implementations from the functions directory"""
        functions = {}
        for tool in tools:
            try:
                module = importlib.import_module(f'functions.{tool["function"]["name"]}')
                functions[tool["function"]["name"]] = getattr(module, tool["function"]["name"])
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load function {tool['function']['name']}: {str(e)}")
        return functions

    def update_context(self, context: CallContext):
        """Update the service context with new information"""
        self.context = context
        self.system_message = context.system_message
        self.initial_message = context.initial_message
        self.medication_schedule = context.schedule

    def log_adherence(self, status: str, dose_time: datetime):
        """Log medication adherence event"""
        self.adherence_log.append({
            "timestamp": dose_time.isoformat(),
            "medication": self.context.medication,
            "dose": self.context.dosage,
            "status": status
        })
        logger.info(f"Adherence logged: {status} for {self.context.medication} at {dose_time.isoformat()}")
        return self.adherence_log

    def calculate_next_dose(self, current_time: datetime) -> Optional[datetime]:
        """Calculate the next scheduled dose time based on the current time"""
        if not self.medication_schedule:
            logger.warning("No medication schedule available")
            return None
            
        schedule = self.medication_schedule
        times = [datetime.strptime(t, "%H:%M").time() for t in schedule.values()]
        current_time = current_time.time()
        
        for t in sorted(times):
            if current_time < t:
                return datetime.combine(datetime.today(), t)
        return datetime.combine(datetime.today() + timedelta(days=1), min(times))

    async def handle_medication_reminder(self, interaction_count: int):
        """Handle sending a medication reminder"""
        next_dose = self.calculate_next_dose(datetime.now())
        if next_dose:
            reminder = f"Time for your {self.context.medication} ({self.context.dosage}). Please confirm you've taken it."
            await self.completion(reminder, interaction_count, role="system")
        else:
            logger.warning("Failed to calculate next dose time")

    @abstractmethod
    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        """Generate a completion response from the LLM"""
        pass

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming"""
        # More comprehensive sentence splitting pattern
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    async def emit_complete_sentences(self, text: str, interaction_count: int):
        """Emit complete sentences as they are generated"""
        self.sentence_buffer += text
        sentences = self.split_into_sentences(self.sentence_buffer)
        
        if not sentences:
            return
            
        for sentence in sentences[:-1]:
            await self.emit('llmreply', {
                "partialResponseIndex": self.partial_response_index,
                "partialResponse": sentence
            }, interaction_count)
            self.partial_response_index += 1
        
        self.sentence_buffer = sentences[-1] if sentences else ""

class OpenAIService(AbstractLLMService):
    def __init__(self, context: CallContext):
        super().__init__(context)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"OpenAI service initialized with model: {self.model}")

    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        """Generate a completion using the OpenAI API"""
        try:
            self.user_context.append({"role": role, "content": text, "name": name})
            messages = [{"role": "system", "content": self.system_message}] + self.user_context

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True,
                max_tokens=500
            )

            complete_response = ""
            tool_responses = []
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    complete_response += content
                    await self.emit_complete_sentences(content, interaction_count)
                
                if chunk.choices[0].delta.tool_calls:
                    tool_call = await self.handle_tool_calls(chunk.choices[0].delta.tool_calls, interaction_count)
                    if tool_call:
                        tool_responses.append(tool_call)

            if complete_response.strip():
                self.user_context.append({"role": "assistant", "content": complete_response})
                
            # Return the final response for logging purposes
            return complete_response

        except Exception as e:
            error_msg = f"OpenAI Error: {str(e)}"
            logger.error(error_msg)
            await self.emit('llmreply', {
                "partialResponseIndex": None,
                "partialResponse": "Sorry, I encountered an error processing your request."
            }, interaction_count)
            return error_msg

    async def handle_tool_calls(self, tool_calls, interaction_count):
        """Handle tool calls from the OpenAI API"""
        for tool in tool_calls:
            try:
                function_name = tool.function.name
                function_args = json.loads(tool.function.arguments)
                
                logger.info(f"Tool call: {function_name} with args: {function_args}")
                
                if function_name in self.available_functions:
                    response = await self.available_functions[function_name](self.context, **function_args)
                    await self.emit('llmreply', {
                        "partialResponseIndex": None,
                        "partialResponse": response.get('confirmation', '')
                    }, interaction_count)
                    return {"function": function_name, "result": response}
                else:
                    logger.warning(f"Function '{function_name}' not found in available functions")
            except Exception as e:
                logger.error(f"Error handling tool call: {str(e)}")
        return None

class AnthropicService(AbstractLLMService):
    def __init__(self, context: CallContext):
        super().__init__(context)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        logger.info(f"Anthropic service initialized with model: {self.model}")

    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        """Generate a completion using the Anthropic API"""
        try:
            self.user_context.append({"role": role, "content": text})
            
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=500,
                system=self.system_message,
                messages=self.user_context,
                tools=self._convert_tools(),
            ) as stream:
                complete_response = ""
                tool_responses = []
                
                async for event in stream:
                    if event.type == "content_block_delta":
                        content = event.delta.text
                        complete_response += content
                        await self.emit_complete_sentences(content, interaction_count)
                    elif event.type == "tool_use":
                        tool_response = await self.handle_tool_use(event, interaction_count)
                        if tool_response:
                            tool_responses.append(tool_response)

                if complete_response.strip():
                    self.user_context.append({"role": "assistant", "content": complete_response})
                
                # Return the final response for logging purposes
                return complete_response

        except Exception as e:
            error_msg = f"Anthropic Error: {str(e)}"
            logger.error(error_msg)
            await self.emit('llmreply', {
                "partialResponseIndex": None,
                "partialResponse": "Sorry, I encountered an error processing your request."
            }, interaction_count)
            return error_msg

    def _convert_tools(self):
        """Convert tools to Anthropic format"""
        return [{
            "name": tool['function']['name'],
            "description": tool['function']['description'],
            "input_schema": tool['function']['parameters']
        } for tool in tools]

    async def handle_tool_use(self, event, interaction_count):
        """Handle tool use events from the Anthropic API"""
        try:
            function_name = event.tool_name
            function_args = event.input
            
            logger.info(f"Tool use: {function_name} with args: {function_args}")
            
            if function_name in self.available_functions:
                response = await self.available_functions[function_name](self.context, **function_args)
                await self.emit('llmreply', {
                    "partialResponseIndex": None,
                    "partialResponse": response.get('confirmation', '')
                }, interaction_count)
                return {"function": function_name, "result": response}
            else:
                logger.warning(f"Function '{function_name}' not found in available functions")
        except Exception as e:
            logger.error(f"Error handling tool use: {str(e)}")
        return None

class GeminiService(AbstractLLMService):
    def __init__(self, context: CallContext):
        super().__init__(context)
        self._configure_gemini()
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=self._convert_tools()
        )
        self.conversation = self.model.start_chat(history=[])
        logger.info(f"Gemini service initialized with model: {self.model_name}")

    def _configure_gemini(self):
        """Initialize Gemini with proper error handling"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing in environment variables")
        genai.configure(api_key=api_key)

    async def completion(self, text: str, interaction_count: int, role: str = 'user', name: str = 'user'):
        """Generate a completion using the Gemini API"""
        try:
            # Build conversation history
            self._update_conversation_history(role, text)
            
            # Generate response with tool handling
            complete_response = ""
            tool_responses = []
            
            async for response in await self.conversation.send_message_async(text, stream=True):
                for part in response.candidates[0].content.parts:
                    if part.text:
                        complete_response += part.text
                        await self.emit_complete_sentences(part.text, interaction_count)
                    elif part.function_call:
                        tool_response = await self._handle_function_call(part.function_call, interaction_count)
                        if tool_response:
                            tool_responses.append(tool_response)

            if complete_response.strip():
                self._update_conversation_history("model", complete_response)

            # Return the final response for logging purposes
            return complete_response

        except Exception as e:
            error_msg = f"Gemini Error: {str(e)}"
            logger.error(error_msg)
            await self.emit('llmreply', {
                "partialResponseIndex": None,
                "partialResponse": "Sorry, I'm having trouble processing your request."
            }, interaction_count)
            return error_msg

    def _update_conversation_history(self, role: str, text: str):
        """Maintain conversation context"""
        self.user_context.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": text}]
        })

    async def _handle_function_call(self, function_call, interaction_count: int):
        """Execute and respond to function calls"""
        try:
            function_name = function_call.name
            function_args = json.loads(function_call.args)
            
            logger.info(f"Function call: {function_name} with args: {function_args}")
            
            if function_name in self.available_functions:
                result = await self.available_functions[function_name](self.context, **function_args)
                
                # Send confirmation to client
                await self.emit('llmreply', {
                    "partialResponseIndex": None,
                    "partialResponse": result.get('confirmation', 'Action completed')
                }, interaction_count)
                
                # Add function result to conversation
                self.conversation.send_message(json.dumps(result))
                return {"function": function_name, "result": result}
            else:
                logger.warning(f"Function '{function_name}' not found in available functions")

        except Exception as e:
            logger.error(f"Function call failed: {str(e)}")
            await self.emit('llmreply', {
                "partialResponseIndex": None,
                "partialResponse": "Failed to execute action"
            }, interaction_count)
        return None

    def _convert_tools(self):
        """Convert OpenAI-style tools to Gemini format"""
        return [genai.Tool(
            function_declarations=[genai.FunctionDeclaration(
                name=tool['function']['name'],
                description=tool['function']['description'],
                parameters=self._adapt_parameters(tool['function']['parameters'])
            )]
        ) for tool in tools]

    def _adapt_parameters(self, params: dict) -> dict:
        """Convert parameters to Gemini-compatible format"""
        return {
            "type_
```"+"```python
def _adapt_parameters(self, params: dict) -> dict:
        """Convert parameters to Gemini-compatible format"""
        return {
            "type_": "object",
            "properties": {
                prop: {"type_": details["type"]} 
                for prop, details in params.get("properties", {}).items()
            },
            "required": params.get("required", [])
        }


class LLMFactory:
    @staticmethod
    def get_service(provider: str, context: CallContext) -> AbstractLLMService:
        """Create and return an LLM service instance based on the provider name"""
        provider = provider.lower()
        try:
            if provider == "openai":
                return OpenAIService(context)
            elif provider == "anthropic":
                return AnthropicService(context)
            elif provider == "gemini":
                return GeminiService(context)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize {provider} service: {str(e)}")
            # Fallback to a default service
            logger.info("Falling back to OpenAI service")
            return OpenAIService(context)

```

```python
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
```"+"```python
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

```

```python
import os
import json
import requests
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
import websockets
import threading
import time

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:3000")

# Set page configuration
st.set_page_config(
    page_title="Smart Medication Manager",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "adherence_data" not in st.session_state:
    st.session_state.adherence_data = []
if "schedule_active" not in st.session_state:
    st.session_state.schedule_active = False
if "current_phone" not in st.session_state:
    st.session_state.current_phone = ""
if "websocket_connected" not in st.session_state:
    st.session_state.websocket_connected = False
if "call_history" not in st.session_state:
    st.session_state.call_history = []
if "last_notification" not in st.session_state:
    st.session_state.last_notification = None
if "next_reminder" not in st.session_state:
    st.session_state.next_reminder = None

# Function to call the backend API
def call_backend_api(endpoint, data=None, method="POST"):
    """Call the backend API with provided data"""
    url = f"{BACKEND_URL}/{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            st.error(f"Unsupported method: {method}")
            return None
            
        if response.status_code in [200, 201, 204]:
            if response.content:
                return response.json()
            return {"status": "success"}
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None

# WebSocket client for real-time updates
async def connect_websocket():
    """Connect to backend WebSocket for real-time updates"""
    if not st.session_state.current_phone:
        return
        
    ws_url = f"{WEBSOCKET_URL}/ws/{st.session_state.current_phone}"
    
    try:
        async with websockets.connect(ws_url) as websocket:
            st.session_state.websocket_connected = True
            
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Handle different message types
                if data["type"] == "adherence_update" or data["type"] == "adherence_logged":
                    # Add new adherence data
                    st.session_state.adherence_data.append(data["data"])
                    # Show notification
                    st.session_state.last_notification = f"Medication {data['data']['status']} at {data['data']['timestamp']}"
                
                elif data["type"] == "schedule":
                    # Update schedule information
                    schedule_data = data["data"]
                    st.session_state.next_reminder = schedule_data.get("next_reminder")
                
                elif data["type"] == "adherence_history":
                    # Replace adherence history
                    st.session_state.adherence_data = data["data"]
                
                elif data["type"] == "reminder_sent":
                    # Update notification
                    reminder_data = data["data"]
                    st.session_state.last_notification = f"Reminder sent for {reminder_data['dose']} dose at {reminder_data['timestamp']}"
                    st.session_state.next_reminder = reminder_data.get("next_reminder")
                
                elif data["type"] == "call_update":
                    # Add to call history
                    call_data = data["data"]
                    st.session_state.call_history.append(call_data)
                    
    except websockets.exceptions.ConnectionClosed:
        st.session_state.websocket_connected = False
    except Exception as e:
        st.error(f"WebSocket error: {str(e)}")
        st.session_state.websocket_connected = False

# Start WebSocket connection in a background thread
def start_websocket_thread():
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            if st.session_state.current_phone and not st.session_state.websocket_connected:
                loop.run_until_complete(connect_websocket())
            time.sleep(1)
    
    thread = threading.Thread(target=run_async_loop, daemon=True)
    thread.start()

# Function to format datetime for display
def format_datetime(dt_str):
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str

# Calculate adherence metrics
def calculate_adherence_metrics(adherence_data):
    if not adherence_data:
        return {
            "total_doses": 0,
            "taken_doses": 0,
            "adherence_rate": 0,
            "streak": 0
        }
    
    total_doses = len(adherence_data)
    taken_doses = sum(1 for log in adherence_data if log.get("status") == "taken")
    adherence_rate = (taken_doses / total_doses) * 100 if total_doses > 0 else 0
    
    # Calculate streak (consecutive days with adherence)
    streak = 0
    dates_with_adherence = set()
    
    for log in adherence_data:
        if log.get("status") == "taken":
            try:
                date_str = datetime.fromisoformat(log["timestamp"]).date().isoformat()
```"  all other codes i got from you but streamlit_app.py code i got half so please give me that