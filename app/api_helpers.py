import json
import time
import math
import asyncio
import base64
import random 
from typing import List, Dict, Any, Callable, Union, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
from google.genai.types import HttpOptions
from google import genai
from openai import AsyncOpenAI # For type hinting
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    deobfuscate_text,
    convert_to_openai_format, # This is our process_gemini_response_to_openai_dict
    convert_chunk_to_openai,  # For true Gemini streaming
    create_final_chunk,
    parse_gemini_response_for_reasoning_and_content, # Used by convert_to_openai_format
    extract_reasoning_by_tags # Used by older OpenAI direct fake streamer
)
import config as app_config
from config import VERTEX_REASONING_TAG

class StreamingReasoningProcessor:
    """Stateful processor for extracting reasoning from streaming content with tags."""
    def __init__(self, tag_name: str = VERTEX_REASONING_TAG):
        self.tag_name = tag_name
        self.open_tag = f"<{tag_name}>"
        self.close_tag = f"</{tag_name}>"
        self.tag_buffer = ""
        self.inside_tag = False
        self.reasoning_buffer = ""
        self.partial_tag_buffer = "" 

    def process_chunk(self, content: str) -> tuple[str, str]:
        if self.partial_tag_buffer:
            content = self.partial_tag_buffer + content
            self.partial_tag_buffer = ""
        self.tag_buffer += content
        processed_content = ""
        current_reasoning = ""
        while self.tag_buffer:
            if not self.inside_tag:
                open_pos = self.tag_buffer.find(self.open_tag)
                if open_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.open_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.open_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                processed_content += self.tag_buffer[:-i]
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        processed_content += self.tag_buffer
                        self.tag_buffer = ""
                    break
                else:
                    processed_content += self.tag_buffer[:open_pos]
                    self.tag_buffer = self.tag_buffer[open_pos + len(self.open_tag):]
                    self.inside_tag = True
            else: # Inside tag
                close_pos = self.tag_buffer.find(self.close_tag)
                if close_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.close_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.close_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                new_reasoning = self.tag_buffer[:-i]
                                self.reasoning_buffer += new_reasoning
                                if new_reasoning: current_reasoning = new_reasoning
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        if self.tag_buffer:
                            self.reasoning_buffer += self.tag_buffer
                            current_reasoning = self.tag_buffer
                            self.tag_buffer = ""
                    break
                else:
                    final_reasoning_chunk = self.tag_buffer[:close_pos]
                    self.reasoning_buffer += final_reasoning_chunk
                    if final_reasoning_chunk: current_reasoning = final_reasoning_chunk
                    self.reasoning_buffer = "" 
                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False
        return processed_content, current_reasoning
    
    def flush_remaining(self) -> tuple[str, str]:
        remaining_content, remaining_reasoning = "", ""
        if self.partial_tag_buffer:
            remaining_content += self.partial_tag_buffer
            self.partial_tag_buffer = ""
        if not self.inside_tag:
            if self.tag_buffer: remaining_content += self.tag_buffer
        else:
            if self.reasoning_buffer: remaining_reasoning = self.reasoning_buffer
            if self.tag_buffer: remaining_content += self.tag_buffer
            self.inside_tag = False
        self.tag_buffer, self.reasoning_buffer = "", ""
        return remaining_content, remaining_reasoning

def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": status_code, "param": None}}

def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config = {}
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    if request.stop is not None: config["stop_sequences"] = request.stop
    if request.seed is not None: config["seed"] = request.seed
    if request.presence_penalty is not None: config["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None: config["frequency_penalty"] = request.frequency_penalty
    if request.n is not None: config["candidate_count"] = request.n
    
    config["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF")
    ]
    config["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

    gemini_tools_list, gemini_tool_config_obj = None, None
    if request.tools:
        function_declarations = []
        for tool_def in request.tools:
            if tool_def.get("type") == "function":
                func_dict = tool_def.get("function", {})
                parameters_schema = func_dict.get("parameters", {})
                try:
                    fd = types.FunctionDeclaration(name=func_dict.get("name", ""), description=func_dict.get("description", ""), parameters=parameters_schema)
                    function_declarations.append(fd)
                except Exception as e: print(f"Error creating FunctionDeclaration for tool {func_dict.get('name', 'unknown')}: {e}")
        if function_declarations: gemini_tools_list = [types.Tool(function_declarations=function_declarations)]

    if request.tool_choice:
        mode_val = types.FunctionCallingConfig.Mode.AUTO 
        allowed_fn_names = None
        if isinstance(request.tool_choice, str):
            if request.tool_choice == "none": mode_val = types.FunctionCallingConfig.Mode.NONE
            elif request.tool_choice == "required": mode_val = types.FunctionCallingConfig.Mode.ANY
        elif isinstance(request.tool_choice, dict) and request.tool_choice.get("type") == "function":
            func_choice_name = request.tool_choice.get("function", {}).get("name")
            if func_choice_name:
                mode_val = types.FunctionCallingConfig.Mode.ANY
                allowed_fn_names = [func_choice_name]
        fcc = types.FunctionCallingConfig(mode=mode_val, allowed_function_names=allowed_fn_names)
        gemini_tool_config_obj = types.ToolConfig(function_calling_config=fcc)

    if gemini_tools_list: config["gemini_tools"] = gemini_tools_list
    if gemini_tool_config_obj: config["gemini_tool_config"] = gemini_tool_config_obj
    return config

def is_gemini_response_valid(response: Any) -> bool:
    if response is None: return False
    if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, 'candidates') and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, 'text') and isinstance(cand.text, str) and cand.text.strip(): return True
            if hasattr(cand, 'content') and hasattr(cand.content, 'parts') and cand.content.parts:
                for part in cand.content.parts:
                    if hasattr(part, 'function_call'): return True 
                    if hasattr(part, 'text') and isinstance(getattr(part, 'text', None), str) and getattr(part, 'text', '').strip(): return True
    return False

async def _chunk_openai_response_dict_for_sse(
    openai_response_dict: Dict[str, Any],
    response_id_override: Optional[str] = None, 
    model_name_override: Optional[str] = None
):
    """Helper to chunk a complete OpenAI-formatted dictionary for SSE."""
    resp_id = response_id_override or openai_response_dict.get("id", f"chatcmpl-fakestream-{int(time.time())}")
    model_name = model_name_override or openai_response_dict.get("model", "unknown")
    created_time = openai_response_dict.get("created", int(time.time()))
    
    choices = openai_response_dict.get("choices", [])
    if not choices: # Should not happen if openai_response_dict is valid
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    for choice_idx, choice in enumerate(choices): # Support multiple choices (n > 1)
        message = choice.get("message", {})
        final_finish_reason = choice.get("finish_reason", "stop")

        if message.get("tool_calls"):
            tool_calls_list = message.get("tool_calls", [])
            for tc_item_idx, tool_call_item in enumerate(tool_calls_list):
                # Delta 1: Tool call structure (name)
                delta_tc_start = {
                    "tool_calls": [{
                        "index": tc_item_idx, # Index of the tool_call in the list
                        "id": tool_call_item["id"],
                        "type": "function",
                        "function": {"name": tool_call_item["function"]["name"], "arguments": ""}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_start, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)

                # Delta 2: Tool call arguments
                delta_tc_args = {
                    "tool_calls": [{
                        "index": tc_item_idx,
                        "id": tool_call_item["id"], # ID can be repeated
                        "function": {"arguments": tool_call_item["function"]["arguments"]}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_args, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)
        
        elif message.get("content") is not None or message.get("reasoning_content") is not None : # Regular content
            reasoning_content = message.get("reasoning_content", "")
            actual_content = message.get("content", "") # Can be None

            if reasoning_content:
                delta_reasoning = {"reasoning_content": reasoning_content}
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_reasoning, 'finish_reason': None}]})}\n\n"
                if actual_content is not None: await asyncio.sleep(0.05)

            content_to_chunk = actual_content if actual_content is not None else ""
            if actual_content is not None:
                chunk_size = max(1, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 1
                if not content_to_chunk and not reasoning_content : # Empty string content
                    yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': ''}, 'finish_reason': None}]})}\n\n"
                else:
                    for i in range(0, len(content_to_chunk), chunk_size):
                        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': content_to_chunk[i:i+chunk_size]}, 'finish_reason': None}]})}\n\n"
                        if len(content_to_chunk) > chunk_size: await asyncio.sleep(0.05)
        
        # Final delta for this choice with finish_reason
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {}, 'finish_reason': final_finish_reason}]})}\n\n"

    yield "data: [DONE]\n\n"


async def gemini_fake_stream_generator( 
    gemini_client_instance: Any, 
    model_for_api_call: str, 
    prompt_for_api_call: Union[types.Content, List[types.Content]],
    gen_config_for_api_call: Dict[str, Any], 
    request_obj: OpenAIRequest,
    is_auto_attempt: bool
):
    model_name_for_log = getattr(gemini_client_instance, 'model_name', 'unknown_gemini_model_object')
    print(f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model string: '{model_for_api_call}', client obj: '{model_name_for_log}')")
    
    internal_tools_param = gen_config_for_api_call.pop('gemini_tools', None)
    internal_tool_config_param = gen_config_for_api_call.pop('gemini_tool_config', None)
    internal_sdk_generation_config = gen_config_for_api_call

    api_call_task = asyncio.create_task(
        gemini_client_instance.aio.models.generate_content(
            model=model_for_api_call, 
            contents=prompt_for_api_call, 
            generation_config=internal_sdk_generation_config,
            tools=internal_tools_param,
            tool_config=internal_tool_config_param
        )
    )

    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)
    
    try:
        raw_gemini_response = await api_call_task 
        openai_response_dict = convert_to_openai_format(raw_gemini_response, request_obj.model)
        
        if hasattr(raw_gemini_response, 'prompt_feedback') and \
           hasattr(raw_gemini_response.prompt_feedback, 'block_reason') and \
           raw_gemini_response.prompt_feedback.block_reason:
            block_message = f"Response blocked by Gemini safety filter: {raw_gemini_response.prompt_feedback.block_reason}"
            if hasattr(raw_gemini_response.prompt_feedback, 'block_reason_message') and \
               raw_gemini_response.prompt_feedback.block_reason_message:
                block_message += f" (Message: {raw_gemini_response.prompt_feedback.block_reason_message})"
            raise ValueError(block_message)

        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict,
            is_auto_attempt=is_auto_attempt # is_auto_attempt is not used by _chunk_openai_response_dict_for_sse directly but good to keep context
        ):
            yield chunk_sse

    except Exception as e_outer_gemini:
        err_msg_detail = f"Error in gemini_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer_gemini).__name__} - {str(e_outer_gemini)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer_gemini)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt: raise


async def openai_fake_stream_generator( 
    openai_client: Union[AsyncOpenAI, Any], # Allow FakeChatCompletion/ExpressClientWrapper
    openai_params: Dict[str, Any],
    openai_extra_body: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool # Though auto-mode is less likely for OpenAI direct path
):
    api_model_name = openai_params.get("model", "unknown-openai-model")
    print(f"FAKE STREAMING (OpenAI Direct): Prep for '{request_obj.model}' (API model: '{api_model_name}')")
    response_id = f"chatcmpl-openaidirectfake-{int(time.time())}"
    
    async def _openai_api_call_task():
        # This call is to an OpenAI-compatible endpoint (Vertex's /openapi)
        # It should return an object that mimics OpenAI's SDK response or can be dumped to a dict.
        params_for_call = openai_params.copy()
        params_for_call['stream'] = False # Ensure non-streaming for the internal call
        return await openai_client.chat.completions.create(**params_for_call, extra_body=openai_extra_body)

    api_call_task = asyncio.create_task(_openai_api_call_task())
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)

    try:
        # raw_response_obj is an OpenAI SDK-like object (e.g. openai.types.chat.ChatCompletion or our FakeChatCompletion)
        raw_response_obj = await api_call_task 
        
        # Convert the OpenAI SDK-like object to a standard dictionary.
        # The .model_dump() method is standard for Pydantic models (which OpenAI SDK uses)
        # and our FakeChatCompletion also implements it.
        openai_response_dict = raw_response_obj.model_dump(exclude_unset=True, exclude_none=True)

        # The Vertex OpenAI endpoint might embed reasoning within the content using tags.
        # If so, extract it. This part is specific to how Vertex /openapi endpoint handles reasoning.
        # If it's a true OpenAI model or an endpoint that doesn't use these tags, this will do nothing.
        if openai_response_dict.get("choices") and \
           openai_response_dict["choices"].get("message", {}).get("content"):
            
            original_content = openai_response_dict["choices"]["message"]["content"]
            # Ensure extract_reasoning_by_tags handles None or non-string gracefully
            if isinstance(original_content, str):
                reasoning_text, actual_content = extract_reasoning_by_tags(original_content, VERTEX_REASONING_TAG)
                openai_response_dict["choices"]["message"]["content"] = actual_content
                if reasoning_text: # Add reasoning_content if found
                    openai_response_dict["choices"]["message"]["reasoning_content"] = reasoning_text
            # If content is not a string (e.g., already None due to tool_calls), skip tag extraction.

        # Now, chunk this openai_response_dict using the common chunking helper
        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict,
            response_id_override=response_id, # Use the one generated for this fake stream
            model_name_override=request_obj.model, # Use the original request model name for SSE
            # is_auto_attempt is not directly used by _chunk_openai_response_dict_for_sse
        ):
            yield chunk_sse
            
    except Exception as e_outer: 
        err_msg_detail = f"Error in openai_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer).__name__} - {str(e_outer)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt: raise


async def execute_gemini_call(
    current_client: Any, 
    model_to_call: str,  
    prompt_func: Callable[[List[OpenAIMessage]], List[types.Content]], 
    gen_config_for_call: Dict[str, Any], 
    request_obj: OpenAIRequest, 
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    client_model_name_for_log = getattr(current_client, 'model_name', 'unknown_direct_client_object')
    print(f"INFO: execute_gemini_call for requested API model '{model_to_call}', using client object with internal name '{client_model_name_for_log}'. Original request model: '{request_obj.model}'")

    # For true streaming and non-streaming, tools/tool_config are passed as top-level args.
    # For fake streaming, gemini_fake_stream_generator will handle extracting them from its gen_config_for_api_call.
    
    if request_obj.stream:
        if app_config.FAKE_STREAMING_ENABLED:
            # Pass the full gen_config_for_call, as gemini_fake_stream_generator
            # will extract gemini_tools and gemini_tool_config internally for its non-streaming call.
            return StreamingResponse(
                gemini_fake_stream_generator(
                    current_client, model_to_call, actual_prompt_for_call,
                    gen_config_for_call.copy(), # Pass a copy to avoid modification issues if any
                    request_obj, is_auto_attempt
                ), media_type="text/event-stream"
            )
        else: # True Streaming
            gemini_tools_param = gen_config_for_call.pop('gemini_tools', None)
            gemini_tool_config_param = gen_config_for_call.pop('gemini_tool_config', None)
            sdk_generation_config = gen_config_for_call # Remainder is for generation_config

            response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"
            async def _gemini_real_stream_generator_inner():
                try:
                    stream_gen_obj = await current_client.aio.models.generate_content_stream(
                        model=model_to_call, contents=actual_prompt_for_call,
                        generation_config=sdk_generation_config,
                        tools=gemini_tools_param, tool_config=gemini_tool_config_param
                    )
                    async for chunk_item_call in stream_gen_obj:
                        yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                    yield "data: [DONE]\n\n"
                except Exception as e_stream_call:
                    err_msg_detail_stream = f"Streaming Error (Gemini API, model string: '{model_to_call}'): {type(e_stream_call).__name__} - {str(e_stream_call)}"
                    print(f"ERROR: {err_msg_detail_stream}")
                    s_err = str(e_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                    err_resp = create_openai_error_response(500,s_err,"server_error")
                    j_err = json.dumps(err_resp)
                    if not is_auto_attempt:
                        yield f"data: {j_err}\n\n"
                        yield "data: [DONE]\n\n"
                    raise e_stream_call
            return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else: # Non-streaming
        gemini_tools_param = gen_config_for_call.pop('gemini_tools', None)
        gemini_tool_config_param = gen_config_for_call.pop('gemini_tool_config', None)
        sdk_generation_config = gen_config_for_call # Remainder

        response_obj_call = await current_client.aio.models.generate_content(
            model=model_to_call, contents=actual_prompt_for_call,
            generation_config=sdk_generation_config,
            tools=gemini_tools_param, tool_config=gemini_tool_config_param
        )
        if hasattr(response_obj_call, 'prompt_feedback') and \
           hasattr(response_obj_call.prompt_feedback, 'block_reason') and \
           response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback,'block_reason_message') and \
               response_obj_call.prompt_feedback.block_reason_message: 
                block_msg+=f" ({response_obj_call.prompt_feedback.block_reason_message})"
            raise ValueError(block_msg)
        
        if not is_gemini_response_valid(response_obj_call):
            error_details = f"Invalid non-streaming Gemini response for model string '{model_to_call}'. "
            # ... (error detail extraction logic remains same)
            if hasattr(response_obj_call, 'candidates'):
                error_details += f"Candidates: {len(response_obj_call.candidates) if response_obj_call.candidates else 0}. "
                if response_obj_call.candidates and len(response_obj_call.candidates) > 0:
                    candidate = response_obj_call.candidates # Check first candidate
                    if hasattr(candidate, 'content'):
                        error_details += "Has content. "
                        if hasattr(candidate.content, 'parts'):
                            error_details += f"Parts: {len(candidate.content.parts) if candidate.content.parts else 0}. "
                            if candidate.content.parts and len(candidate.content.parts) > 0:
                                part = candidate.content.parts # Check first part
                                if hasattr(part, 'text'):
                                    text_preview = str(getattr(part, 'text', ''))[:100]
                                    error_details += f"First part text: '{text_preview}'"
                                elif hasattr(part, 'function_call'):
                                    error_details += f"First part is function_call: {part.function_call.name}"

            else:
                error_details += f"Response type: {type(response_obj_call).__name__}"
            raise ValueError(error_details)
        return JSONResponse(content=convert_to_openai_format(response_obj_call, request_obj.model))