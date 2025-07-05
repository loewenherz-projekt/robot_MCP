"""
Base LLM Provider interface and response classes.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = None
    provider: str = ""
    usage: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.usage is None:
            self.usage = {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Return whether the provider supports thinking."""
        pass
    
    def format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to LLM format. Override if provider needs special formatting."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}})
            }
            for tool in tools
        ]
    
    def format_tool_calls_for_execution(self, raw_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw tool calls from LLM for local execution. Override if needed."""
        formatted_tool_calls = []
        for tool_call in raw_tool_calls:
            try:
                arguments = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            
            formatted_tool_calls.append({
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": arguments
            })
        
        return formatted_tool_calls
    
    @abstractmethod
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for the provider's API."""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for the provider's API."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        thinking_enabled: bool = False,
        thinking_budget: int = 1024,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    def print_thinking_header(self):
        """Print thinking header."""
        print("🧠 Thinking: ", end="", flush=True)
    
    def print_response_header(self):
        """Print response header."""
        print(f"🤖 {self.provider_name}: ", end="", flush=True)
    
    def format_tool_results_for_conversation(self, tool_calls: List[Dict[str, Any]], tool_outputs: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format tool results for conversation history. Override if needed."""
        tool_results_with_images = []
        image_parts = []
        
        for tool_call, tool_output_parts in zip(tool_calls, tool_outputs):
            text_parts = []
            current_image_parts = []

            for part in tool_output_parts:
                if part['type'] == 'image':
                    current_image_parts.append(part)
                    image_parts.append(part)
                else:
                    text_parts.append(part.get('text', str(part)))
            
            tool_results_with_images.append({
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": "\n".join(text_parts) if text_parts else "Tool executed successfully."
            })
            
            for i, image_part in enumerate(current_image_parts, 1):
                tool_results_with_images.append({
                    "type": "text",
                    "text": f"Image {i}:"
                })
                tool_results_with_images.append(image_part)
        
        return tool_results_with_images, image_parts 