"""LLM client utilities."""

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from openai import OpenAI

from .logger import get_logger
from .structures import LLMResponse


class LLMClient:
    """
    Thin wrapper over an OpenAI-compatible chat-completions client.

    Features:
    - Retry handling
    - Optional JSON mode
    - Per-step call logging
    - Support for arbitrary extra API parameters
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4",
        timeout: int = 120,
        retry_times: int = 3,
        retry_delay: int = 5,
        **extra_params,
    ):
        """
        Args:
            api_key: API key.
            base_url: API base URL.
            model: Default model.
            timeout: Request timeout in seconds.
            retry_times: Number of retry attempts.
            retry_delay: Delay between retries in seconds.
            **extra_params: Extra API parameters such as temperature or max_tokens.
        """
        self.model = model
        self.timeout = timeout
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.extra_params = extra_params

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        self.logger = get_logger()
        self._thread_local = threading.local()

    def set_log_dir(self, log_dir: Optional[Path]):
        """
        Set the log directory for the current worker/thread.

        Args:
            log_dir: Directory to store LLM call logs, or `None` to disable.
        """
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._thread_local.log_dir = log_dir
            self._thread_local.call_counter = 0
        else:
            self._thread_local.log_dir = None
            self._thread_local.call_counter = 0

    def _get_log_dir(self) -> Optional[Path]:
        """Return the active thread-local log directory."""
        return getattr(self._thread_local, "log_dir", None)

    def _get_call_counter(self) -> int:
        """Return the active thread-local call counter."""
        return getattr(self._thread_local, "call_counter", 0)

    def _increment_call_counter(self):
        """Increment the thread-local call counter."""
        if not hasattr(self._thread_local, "call_counter"):
            self._thread_local.call_counter = 0
        self._thread_local.call_counter += 1

    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        model: Optional[str] = None,
        call_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat-completions request.

        Args:
            messages: Chat messages.
            json_mode: Whether to request a JSON object response.
            model: Optional model override.
            call_name: Optional label for call logging.
            **kwargs: Request-level overrides.

        Returns:
            Structured `LLMResponse`.
        """
        params = {
            "model": model or self.model,
            "messages": messages,
            **self.extra_params,
            **kwargs,
        }

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.retry_times):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                call_time = time.time() - start_time

                content = response.choices[0].message.content or ""

                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                        "total_tokens": response.usage.total_tokens or 0,
                    }

                result = LLMResponse(
                    content=content,
                    raw_response=response,
                    usage=usage,
                    model=params["model"],
                    call_time=call_time,
                )

                self.logger.log_llm_call({
                    "model": params["model"],
                    "usage": usage,
                    "call_time": call_time,
                })

                log_dir = self._get_log_dir()
                if log_dir:
                    self._log_call_to_file(messages, result, call_name, attempt)

                return result

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.retry_times}): {e}"
                )
                if attempt < self.retry_times - 1:
                    time.sleep(self.retry_delay)

        raise last_error

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        call_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Convenience wrapper for a single user prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            json_mode: Whether to request JSON output.
            call_name: Optional label for call logging.
            **kwargs: Additional request overrides.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, json_mode=json_mode, call_name=call_name, **kwargs)

    def extract_tags(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        call_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Extract XML-like tags from an LLM response.

        Example:
            <name>...</name>
            <motivation>...</motivation>
        """
        response = self.generate(prompt, system_prompt, json_mode=False, call_name=call_name, **kwargs)
        content = response.content.strip()

        result = {}

        tag_pattern = r"<(\w+)>"
        pos = 0
        while True:
            match = re.search(tag_pattern, content[pos:])
            if not match:
                break

            tag_name = match.group(1)
            tag_start = pos + match.end()

            end_tag = f"</{tag_name}>"
            end_pos = content.find(end_tag, tag_start)

            if end_pos == -1:
                pos = tag_start
                continue

            tag_content = content[tag_start:end_pos].strip()
            result[tag_name] = tag_content
            pos = end_pos + len(end_tag)

        if not result:
            self.logger.error("Failed to extract tags from response")
            self.logger.error(f"Response content (full, {len(content)} chars):\n{content[:1000]}...")
            raise ValueError("No valid tags found in LLM response")

        self.logger.debug(f"Extracted {len(result)} tags: {list(result.keys())}")
        return result

    def _log_call_to_file(
        self,
        messages: List[Dict[str, str]],
        response: LLMResponse,
        call_name: Optional[str],
        attempt: int,
    ):
        """Persist one LLM call to a JSON log file."""
        log_dir = self._get_log_dir()
        if not log_dir:
            return

        self._increment_call_counter()
        call_counter = self._get_call_counter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        if call_name:
            filename = f"llm_call_{call_counter:03d}_{call_name}_{timestamp}.json"
        else:
            filename = f"llm_call_{call_counter:03d}_{timestamp}.json"

        log_file = log_dir / filename

        log_data = {
            "call_number": call_counter,
            "timestamp": datetime.now().isoformat(),
            "call_name": call_name,
            "attempt": attempt + 1,
            "model": response.model,
            "usage": response.usage,
            "call_time": response.call_time,
            "messages": messages,
            "response": {
                "content": response.content,
                "finish_reason": response.raw_response.choices[0].finish_reason if response.raw_response.choices else None,
            },
        }

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write LLM call log: {e}")


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """Create an `LLMClient` from the top-level config dictionary."""
    api_config = config.get("api", {})

    framework_keys = {"provider", "base_url", "api_key", "model", "timeout", "retry_times", "retry_delay"}

    framework_params = {
        "api_key": api_config.get("api_key", "EMPTY"),
        "base_url": api_config.get("base_url", "https://api.openai.com/v1"),
        "model": api_config.get("model", "default"),
        "timeout": api_config.get("timeout", 120),
        "retry_times": api_config.get("retry_times", 3),
        "retry_delay": api_config.get("retry_delay", 5),
    }

    extra_params = {k: v for k, v in api_config.items() if k not in framework_keys}

    return LLMClient(**framework_params, **extra_params)
