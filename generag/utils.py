"""Utility functions for Gene RAG system."""

import json
import re
import requests
import time
from typing import Any, Dict, Optional

from .config import Config


def call_llm(
    prompt: str,
    config: Config,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: Optional[int] = None
) -> str:
    """Call LLM API with retry mechanism.
    
    Args:
        prompt: User prompt
        config: Configuration object
        system_message: System message for LLM
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        max_retries: Maximum number of retries (overrides config if provided)
        
    Returns:
        LLM response text
        
    Raises:
        Exception: If all retries fail
    """
    max_retries = max_retries or getattr(config, 'max_retries', 3)
    
    for attempt in range(max_retries):
        try:
            return _call_llm_internal(prompt, config, system_message, temperature, max_tokens)
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
            
            # Exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)


def _call_llm_internal(
    prompt: str,
    config: Config,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    """Internal LLM API call without retry.
    
    Args:
        prompt: User prompt
        config: Configuration object
        system_message: System message for LLM
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        LLM response text
        
    Raises:
        Exception: If API call fails
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    data = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            config.api_url, 
            headers=headers, 
            json=data, 
            timeout=getattr(config, 'timeout', 120)
        )
        response.raise_for_status()
        result = response.json()
        
        if "choices" not in result or not result["choices"]:
            raise Exception("Invalid response format from API")
            
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        raise Exception(f"LLM API call timed out after {getattr(config, 'timeout', 120)} seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM API request failed: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {e}")
    except Exception as e:
        raise Exception(f"LLM API call failed: {e}")


def parse_json_response(response: str) -> Any:
    """Parse JSON from LLM response.
    
    Args:
        response: LLM response text
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If JSON parsing fails
    """
    # Try to extract JSON from response
    response = response.strip()
    
    # Remove markdown code blocks if present
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    
    # Try to find JSON object or array
    json_match = re.search(r'[\[{].*[\]}]', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    # Try direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("No valid JSON found in response")