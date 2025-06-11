import json
import time
import math
import asyncio
import base64
from typing import List, Dict, Any, Callable, Union, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
from google.genai.types import HttpOptions
from google import genai # Original import
from openai import AsyncOpenAI

from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    deobfuscate_text,
    convert_to_openai_format,
    convert_chunk_to_openai,
    create_final_chunk,
    parse_gemini_response_for_reasoning_and_content, # Added import
    extract_reasoning_by_tags # Added for new OpenAI direct reasoning logic
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
        self.partial_tag_buffer = ""  # Buffer for potential partial tags
    
    def process_chunk(self, content: str) -> tuple[str, str]:
        """
        Process a chunk of streaming content.
        
        Args:
            content: New content from the stream
            
        Returns:
            A tuple of:
            - processed_content: Content with reasoning tags removed
            - current_reasoning: Reasoning text found in this chunk (partial or complete)
        """
        # Add new content to buffer, but also handle any partial tag from before
        if self.partial_tag_buffer:
            # We had a partial tag from the previous chunk
            content = self.partial_tag_buffer + content
            self.partial_tag_buffer = ""
        
        self.tag_buffer += content
        
        processed_content = ""
        current_reasoning = ""
        
        while self.tag_buffer:
            if not self.inside_tag:
                # Look for opening tag
                open_pos = self.tag_buffer.find(self.open_tag)
                if open_pos == -1:
                    # No complete opening tag found
                    # Check if we might have a partial tag at the end
                    partial_match = False
                    for i in range(1, min(len(self.open_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.open_tag[:i]:
                            partial_match = True
                            # Output everything except the potential partial tag
                            if len(self.tag_buffer) > i:
                                processed_content += self.tag_buffer[:-i]
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                                self.tag_buffer = ""
                            else:
                                # Entire buffer is partial tag
                                self.partial_tag_buffer = self.tag_buffer
                                self.tag_buffer = ""
                            break
                    
                    if not partial_match:
                        # No partial tag, output everything
                        processed_content += self.tag_buffer
                        self.tag_buffer = ""
                    break
                else:
                    # Found opening tag
                    processed_content += self.tag_buffer[:open_pos]
                    self.tag_buffer = self.tag_buffer[open_pos + len(self.open_tag):]
                    self.inside_tag = True
            else:
                # Inside tag, look for closing tag
                close_pos = self.tag_buffer.find(self.close_tag)
                if close_pos == -1:
                    # No complete closing tag yet
                    # Check for partial closing tag
                    partial_match = False
                    for i in range(1, min(len(self.close_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.close_tag[:i]:
                            partial_match = True
                            # Add everything except potential partial tag to reasoning
                            if len(self.tag_buffer) > i:
                                new_reasoning = self.tag_buffer[:-i]
                                self.reasoning_buffer += new_reasoning
                                if new_reasoning:  # Stream reasoning as it arrives
                                    current_reasoning = new_reasoning
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                                self.tag_buffer = ""
                            else:
                                # Entire buffer is partial tag
                                self.partial_tag_buffer = self.tag_buffer
                                self.tag_buffer = ""
                            break
                    
                    if not partial_match:
                        # No partial tag, add all to reasoning and stream it
                        if self.tag_buffer:
                            self.reasoning_buffer += self.tag_buffer
                            current_reasoning = self.tag_buffer
                            self.tag_buffer = ""
                    break
                else:
                    # Found closing tag
                    final_reasoning_chunk = self.tag_buffer[:close_pos]
                    self.reasoning_buffer += final_reasoning_chunk
                    if final_reasoning_chunk:  # Include the last chunk of reasoning
                        current_reasoning = final_reasoning_chunk
                    self.reasoning_buffer = ""  # Clear buffer after complete tag
                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False
        
        return processed_content, current_reasoning
    
    def flush_remaining(self) -> tuple[str, str]:
        """
        Flush any remaining content in the buffer when the stream ends.
        
        Returns:
            A tuple of:
            - remaining_content: Any content that was buffered but not yet output
            - remaining_reasoning: Any incomplete reasoning if we were inside a tag
        """
        remaining_content = ""
        remaining_reasoning = ""
        
        # First handle any partial tag buffer
        if self.partial_tag_buffer:
            # The partial tag wasn't completed, so treat it as regular content
            remaining_content += self.partial_tag_buffer
            self.partial_tag_buffer = ""
        
        if not self.inside_tag:
            # If we're not inside a tag, output any remaining buffer
            if self.tag_buffer:
                remaining_content += self.tag_buffer
                self.tag_buffer = ""
        else:
            # If we're inside a tag when stream ends, we have incomplete reasoning
            # First, yield any reasoning we've accumulated
            if self.reasoning_buffer:
                remaining_reasoning = self.reasoning_buffer
                self.reasoning_buffer = ""
            
            # Then output the remaining buffer as content (it's an incomplete tag)
            if self.tag_buffer:
                # Don't include the opening tag in output - just the buffer content
                remaining_content += self.tag_buffer
                self.tag_buffer = ""
            
            self.inside_tag = False
        
        return remaining_content, remaining_reasoning


def process_streaming_content_with_reasoning_tags(
    content: str,
    tag_buffer: str,
    inside_tag: bool,
    reasoning_buffer: str,
    tag_name: str = VERTEX_REASONING_TAG
) -> tuple[str, str, bool, str, str]:
    """
    Process streaming content to extract reasoning within tags.
    
    This is a compatibility wrapper for the stateful function. Consider using
    StreamingReasoningProcessor class directly for cleaner code.
    
    Args:
        content: New content from the stream
        tag_buffer: Existing buffer for handling tags split across chunks
        inside_tag: Whether we're currently inside a reasoning tag
        reasoning_buffer: Buffer for accumulating reasoning content
        tag_name: The tag name to look for (defaults to VERTEX_REASONING_TAG)
    
    Returns:
        A tuple of:
        - processed_content: Content with reasoning tags removed
        - current_reasoning: Complete reasoning text if a closing tag was found
        - inside_tag: Updated state of whether we're inside a tag
        - reasoning_buffer: Updated reasoning buffer
        - tag_buffer: Updated tag buffer
    """
    # Create a temporary processor with the current state
    processor = StreamingReasoningProcessor(tag_name)
    processor.tag_buffer = tag_buffer
    processor.inside_tag = inside_tag
    processor.reasoning_buffer = reasoning_buffer
    
    # Process the chunk
    processed_content, current_reasoning = processor.process_chunk(content)
    
    # Return the updated state
    return (processed_content, current_reasoning, processor.inside_tag,
            processor.reasoning_buffer, processor.tag_buffer)

def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
            "param": None,
        }
    }

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
    return config

def is_gemini_response_valid(response: Any) -> bool:
    if response is None: return False
    
    # Check for direct text attribute (SDK response)
    if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip():
        return True
    
    # Check for candidates in the response
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            # Check for direct text on candidate
            if hasattr(candidate, 'text') and isinstance(candidate.text, str) and candidate.text.strip():
                return True
            
            # Check for content with parts
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                for part_item in candidate.content.parts:
                    # Check if part has text (handle both SDK and AttrDict)
                    if hasattr(part_item, 'text'):
                        # AttrDict might have empty string instead of None
                        part_text = getattr(part_item, 'text', None)
                        if part_text is not None and isinstance(part_text, str) and part_text.strip():
                            return True
    
    return False

async def _base_fake_stream_engine(
    api_call_task_creator: Callable[[], asyncio.Task],
    extract_text_from_response_func: Callable[[Any], str], 
    response_id: str,
    sse_model_name: str,
    is_auto_attempt: bool,
    is_valid_response_func: Callable[[Any], bool],
    keep_alive_interval_seconds: float, 
    process_text_func: Optional[Callable[[str, str], str]] = None, 
    check_block_reason_func: Optional[Callable[[Any], None]] = None,
    reasoning_text_to_yield: Optional[str] = None,
    actual_content_text_to_yield: Optional[str] = None
):
    api_call_task = api_call_task_creator()

    if keep_alive_interval_seconds > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"delta": {"reasoning_content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(keep_alive_interval_seconds) 
    
    try:
        full_api_response = await api_call_task 

        if check_block_reason_func:
            check_block_reason_func(full_api_response) 

        if not is_valid_response_func(full_api_response): 
             raise ValueError(f"Invalid/empty API response in fake stream for model {sse_model_name}: {str(full_api_response)[:200]}")

        final_reasoning_text = reasoning_text_to_yield
        final_actual_content_text = actual_content_text_to_yield

        if final_reasoning_text is None and final_actual_content_text is None:
            extracted_full_text = extract_text_from_response_func(full_api_response)
            if process_text_func:
                final_actual_content_text = process_text_func(extracted_full_text, sse_model_name)
            else:
                final_actual_content_text = extracted_full_text
        else:
            if process_text_func:
                if final_reasoning_text is not None:
                    final_reasoning_text = process_text_func(final_reasoning_text, sse_model_name)
                if final_actual_content_text is not None:
                    final_actual_content_text = process_text_func(final_actual_content_text, sse_model_name)
        
        if final_reasoning_text: 
            reasoning_delta_data = {
                "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": sse_model_name, "choices": [{"index": 0, "delta": {"reasoning_content": final_reasoning_text}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(reasoning_delta_data)}\n\n"
            if final_actual_content_text: 
                await asyncio.sleep(0.05) 

        content_to_chunk = final_actual_content_text or "" 
        chunk_size = max(20, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 0
        
        if not content_to_chunk and content_to_chunk != "": 
            empty_delta_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]}
            yield f"data: {json.dumps(empty_delta_data)}\n\n"
        else: 
            for i in range(0, len(content_to_chunk), chunk_size):
                chunk_text = content_to_chunk[i:i+chunk_size]
                content_delta_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]}
                yield f"data: {json.dumps(content_delta_data)}\n\n"
                if len(content_to_chunk) > chunk_size: await asyncio.sleep(0.05)

        yield create_final_chunk(sse_model_name, response_id)
        yield "data: [DONE]\n\n"

    except Exception as e:
        err_msg_detail = f"Error in _base_fake_stream_engine (model: '{sse_model_name}'): {type(e).__name__} - {str(e)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e) 
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_for_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_for_fake_stream_error = json.dumps(err_resp_for_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_for_fake_stream_error}\n\n"
            yield "data: [DONE]\n\n"
        raise

async def gemini_fake_stream_generator( # Changed to async
    gemini_client_instance: Any, 
    model_for_api_call: str, 
    prompt_for_api_call: Union[types.Content, List[types.Content]],
    gen_config_for_api_call: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool
):
    model_name_for_log = getattr(gemini_client_instance, 'model_name', 'unknown_gemini_model_object')
    print(f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model string: '{model_for_api_call}', client obj: '{model_name_for_log}') with reasoning separation.")
    response_id = f"chatcmpl-{int(time.time())}"

    # 1. Create and await the API call task
    api_call_task = asyncio.create_task(
        gemini_client_instance.aio.models.generate_content(
            model=model_for_api_call, 
            contents=prompt_for_api_call, 
            config=gen_config_for_api_call
        )
    )

    # Keep-alive loop while the main API call is in progress
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"reasoning_content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)
    
    try:
        raw_response = await api_call_task # Get the full Gemini response

        # 2. Parse the response for reasoning and content using the centralized parser
        separated_reasoning_text = ""
        separated_actual_content_text = ""
        if hasattr(raw_response, 'candidates') and raw_response.candidates:
            # Typically, fake streaming would focus on the first candidate
            separated_reasoning_text, separated_actual_content_text = parse_gemini_response_for_reasoning_and_content(raw_response.candidates[0])
        elif hasattr(raw_response, 'text') and raw_response.text is not None: # Fallback for simpler response structures
             separated_actual_content_text = raw_response.text


        # 3. Define a text processing function (e.g., for deobfuscation)
        def _process_gemini_text_if_needed(text: str, model_name: str) -> str:
            if model_name.endswith("-encrypt-full"):
                return deobfuscate_text(text)
            return text

        final_reasoning_text = _process_gemini_text_if_needed(separated_reasoning_text, request_obj.model)
        final_actual_content_text = _process_gemini_text_if_needed(separated_actual_content_text, request_obj.model)

        # Define block checking for the raw response
        def _check_gemini_block_wrapper(response_to_check: Any):
            if hasattr(response_to_check, 'prompt_feedback') and hasattr(response_to_check.prompt_feedback, 'block_reason') and response_to_check.prompt_feedback.block_reason:
                block_message = f"Response blocked by Gemini safety filter: {response_to_check.prompt_feedback.block_reason}"
                if hasattr(response_to_check.prompt_feedback, 'block_reason_message') and response_to_check.prompt_feedback.block_reason_message:
                    block_message += f" (Message: {response_to_check.prompt_feedback.block_reason_message})"
                raise ValueError(block_message)

        # Call _base_fake_stream_engine with pre-split and processed texts
        async for chunk in _base_fake_stream_engine(
            api_call_task_creator=lambda: asyncio.create_task(asyncio.sleep(0, result=raw_response)), # Dummy task
            extract_text_from_response_func=lambda r: "", # Not directly used as text is pre-split
            is_valid_response_func=is_gemini_response_valid, # Validates raw_response
            check_block_reason_func=_check_gemini_block_wrapper, # Checks raw_response
            process_text_func=None, # Text processing already done above
            response_id=response_id, 
            sse_model_name=request_obj.model,
            keep_alive_interval_seconds=0, # Keep-alive for this inner call is 0
            is_auto_attempt=is_auto_attempt,
            reasoning_text_to_yield=final_reasoning_text,
            actual_content_text_to_yield=final_actual_content_text
        ):
            yield chunk

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
        # Consider re-raising if auto-mode needs to catch this: raise e_outer_gemini


async def openai_fake_stream_generator( # Reverted signature: removed thought_tag_marker
    openai_client: AsyncOpenAI,
    openai_params: Dict[str, Any],
    openai_extra_body: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool
    # Removed thought_tag_marker as parsing uses a fixed tag now
    # Removed gcp_credentials, gcp_project_id, gcp_location, base_model_id_for_tokenizer previously
):
    api_model_name = openai_params.get("model", "unknown-openai-model")
    print(f"FAKE STREAMING (OpenAI): Prep for '{request_obj.model}' (API model: '{api_model_name}') with reasoning split.")
    response_id = f"chatcmpl-{int(time.time())}"
    
    async def _openai_api_call_and_split_task_creator_wrapper():
        params_for_non_stream_call = openai_params.copy()
        params_for_non_stream_call['stream'] = False
        
        # Use the already configured extra_body which includes the thought_tag_marker
        _api_call_task = asyncio.create_task(
            openai_client.chat.completions.create(**params_for_non_stream_call, extra_body=openai_extra_body)
        )
        raw_response = await _api_call_task
        full_content_from_api = ""
        if raw_response.choices and raw_response.choices[0].message and raw_response.choices[0].message.content is not None:
            full_content_from_api = raw_response.choices[0].message.content
        vertex_completion_tokens = 0
        if raw_response.usage and raw_response.usage.completion_tokens is not None:
            vertex_completion_tokens = raw_response.usage.completion_tokens
        # --- Start Inserted Block (Tag-based reasoning extraction) ---
        reasoning_text = ""
        # Ensure actual_content_text is a string even if API returns None
        actual_content_text = full_content_from_api if isinstance(full_content_from_api, str) else ""

        if actual_content_text: # Check if content exists
            print(f"INFO: OpenAI Direct Fake-Streaming - Applying tag extraction with fixed marker: '{VERTEX_REASONING_TAG}'")
            # Unconditionally attempt extraction with the fixed tag
            reasoning_text, actual_content_text = extract_reasoning_by_tags(actual_content_text, VERTEX_REASONING_TAG)
            # if reasoning_text:
            #      print(f"DEBUG: Tag extraction success (fixed tag). Reasoning len: {len(reasoning_text)}, Content len: {len(actual_content_text)}")
            # else:
            #      print(f"DEBUG: No content found within fixed tag '{VERTEX_REASONING_TAG}'.")
        else:
             print(f"WARNING: OpenAI Direct Fake-Streaming - No initial content found in message.")
             actual_content_text = "" # Ensure empty string

        # --- End Revised Block ---

        # The return uses the potentially modified variables:
        return raw_response, reasoning_text, actual_content_text

    temp_task_for_keepalive_check = asyncio.create_task(_openai_api_call_and_split_task_creator_wrapper())
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not temp_task_for_keepalive_check.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)

    try:
        full_api_response, separated_reasoning_text, separated_actual_content_text = await temp_task_for_keepalive_check
        def _extract_openai_full_text(response: Any) -> str: 
            if response.choices and response.choices[0].message and response.choices[0].message.content is not None:
                return response.choices[0].message.content
            return ""
        def _is_openai_response_valid(response: Any) -> bool:
            return bool(response.choices and response.choices[0].message is not None)

        async for chunk in _base_fake_stream_engine(
            api_call_task_creator=lambda: asyncio.create_task(asyncio.sleep(0, result=full_api_response)), 
            extract_text_from_response_func=_extract_openai_full_text, 
            is_valid_response_func=_is_openai_response_valid,
            response_id=response_id,
            sse_model_name=request_obj.model,
            keep_alive_interval_seconds=0, 
            is_auto_attempt=is_auto_attempt,
            reasoning_text_to_yield=separated_reasoning_text,
            actual_content_text_to_yield=separated_actual_content_text
        ):
            yield chunk
            
    except Exception as e_outer: 
        err_msg_detail = f"Error in openai_fake_stream_generator outer (model: '{request_obj.model}'): {type(e_outer).__name__} - {str(e_outer)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"

async def execute_gemini_call(
    current_client: Any, 
    model_to_call: str,  
    prompt_func: Callable[[List[OpenAIMessage]], Union[types.Content, List[types.Content]]], 
    gen_config_for_call: Dict[str, Any], 
    request_obj: OpenAIRequest, 
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    client_model_name_for_log = getattr(current_client, 'model_name', 'unknown_direct_client_object')
    print(f"INFO: execute_gemini_call for requested API model '{model_to_call}', using client object with internal name '{client_model_name_for_log}'. Original request model: '{request_obj.model}'")

    if request_obj.stream:
        if app_config.FAKE_STREAMING_ENABLED:
            return StreamingResponse(
                gemini_fake_stream_generator( 
                    current_client, 
                    model_to_call, 
                    actual_prompt_for_call, 
                    gen_config_for_call, 
                    request_obj, 
                    is_auto_attempt
                ), 
                media_type="text/event-stream"
            )
        
        response_id_for_stream = f"chatcmpl-{int(time.time())}"
        cand_count_stream = request_obj.n or 1
        
        async def _gemini_real_stream_generator_inner():
            try:
                async for chunk_item_call in await current_client.aio.models.generate_content_stream(
                    model=model_to_call, 
                    contents=actual_prompt_for_call, 
                    config=gen_config_for_call
                ):
                    yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                yield create_final_chunk(request_obj.model, response_id_for_stream, cand_count_stream)
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
    else: 
        response_obj_call = await current_client.aio.models.generate_content(
            model=model_to_call, 
            contents=actual_prompt_for_call, 
            config=gen_config_for_call
        )
        if hasattr(response_obj_call, 'prompt_feedback') and hasattr(response_obj_call.prompt_feedback, 'block_reason') and response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback,'block_reason_message') and response_obj_call.prompt_feedback.block_reason_message: 
                block_msg+=f" ({response_obj_call.prompt_feedback.block_reason_message})"
            raise ValueError(block_msg)
        
        if not is_gemini_response_valid(response_obj_call):
            # Create a more informative error message
            error_details = f"Invalid non-streaming Gemini response for model string '{model_to_call}'. "
            
            # Try to extract useful information from the response
            if hasattr(response_obj_call, 'candidates'):
                error_details += f"Candidates: {len(response_obj_call.candidates) if response_obj_call.candidates else 0}. "
                if response_obj_call.candidates and len(response_obj_call.candidates) > 0:
                    candidate = response_obj_call.candidates[0]
                    if hasattr(candidate, 'content'):
                        error_details += "Has content. "
                        if hasattr(candidate.content, 'parts'):
                            error_details += f"Parts: {len(candidate.content.parts) if candidate.content.parts else 0}. "
                            if candidate.content.parts and len(candidate.content.parts) > 0:
                                part = candidate.content.parts[0]
                                if hasattr(part, 'text'):
                                    text_preview = str(getattr(part, 'text', ''))[:100]
                                    error_details += f"First part text: '{text_preview}'"
            else:
                # If it's not the expected structure, show the type
                error_details += f"Response type: {type(response_obj_call).__name__}"
            
            raise ValueError(error_details)
        return JSONResponse(content=convert_to_openai_format(response_obj_call, request_obj.model))