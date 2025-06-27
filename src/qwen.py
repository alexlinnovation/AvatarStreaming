import asyncio
import json
import base64
import io
import wave
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime
import uuid
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import librosa
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import functools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qwen Omni integration
class QwenOmniClient:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Qwen Omni model and processor"""
        try:
            logger.info("Loading Qwen Omni model...")
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            )
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading Qwen Omni model: {e}")
            raise e
    
    def _save_audio_to_temp_file(self, audio_data: bytes) -> str:
        """Save audio bytes to a temporary file and return the path"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    
    def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio data to the required format"""
        try:
            # Save audio data to temporary file
            temp_path = self._save_audio_to_temp_file(audio_data)
            
            # Load audio using librosa
            audio, sr = librosa.load(temp_path, sr=16000)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise e
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """
        Convert speech to text using Qwen Omni
        """
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_data)
            
            # Create messages for speech-to-text
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a speech recognition model."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Transcribe the audio into text."},
                ]},
            ]
            
            # Run inference
            result = self._run_inference(messages, audio_array=audio)
            return result
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    def generate_response(self, text: str, conversation_history: List[Dict]) -> str:
        """
        Generate response using Qwen Omni LLM
        """
        try:
            # Create system message
            system_prompt = "You are Qwen, a helpful AI assistant created by Alibaba Cloud. You are knowledgeable, helpful, and honest."
            
            # Build conversation messages
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]
            
            # Add conversation history (last 10 messages to avoid context overflow)
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            
            for msg in recent_history[:-1]:  # Exclude the last message as it's the current input
                messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })
            
            # Run inference
            result = self._run_inference(messages)
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _run_inference(self, messages: List[Dict], audio_array: np.ndarray = None) -> str:
        """
        Run inference with the Qwen Omni model
        """
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process multimedia information
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            # If we have audio_array, use it instead
            if audio_array is not None:
                audios = [audio_array] if audios is None or len(audios) == 0 else audios
            
            # Process inputs
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=True
            )
            
            # Move inputs to model device
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, 
                    use_audio_in_video=True, 
                    return_audio=False, 
                    thinker_max_new_tokens=256, 
                    thinker_do_sample=False,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Decode output
            response = self.processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Extract the response (remove the input part)
            full_response = response[0]
            # Find the assistant's response part
            if "<|im_start|>assistant\n" in full_response:
                assistant_response = full_response.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in assistant_response:
                    assistant_response = assistant_response.split("<|im_end|>")[0]
                return assistant_response.strip()
            else:
                # Fallback: return the last part after the last user message
                return full_response.split("user\n")[-1].strip()
            
        except Exception as e:
            logger.error(f"Error in _run_inference: {e}")
            raise e
