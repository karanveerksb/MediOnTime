#llm_service.py
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