"""
Unit tests for LLM providers.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
from typing import Dict, List, Any
import asyncio

from llm_providers.base_provider import LLMProvider, LLMResponse
from llm_providers.factory import create_llm_provider
from llm_providers.claude_provider import ClaudeProvider
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.openai_provider import OpenAIProvider

# Import for type checking
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class TestLLMResponse(unittest.TestCase):
    """Test LLMResponse dataclass."""
    
    def test_default_initialization(self):
        """Test LLMResponse with default values."""
        response = LLMResponse()
        
        self.assertIsNone(response.content)
        self.assertIsNone(response.thinking)
        self.assertIsNone(response.tool_calls)
        self.assertEqual(response.provider, "")
        self.assertEqual(response.usage, {})
    
    def test_full_initialization(self):
        """Test LLMResponse with all values."""
        response = LLMResponse(
            content="Test content",
            thinking="Test thinking",
            tool_calls=[{"id": "1", "name": "test"}],
            provider="test_provider",
            usage={"tokens": 100}
        )
        self.assertEqual(response.content, "Test content")
        self.assertEqual(response.thinking, "Test thinking")
        self.assertEqual(response.tool_calls, [{"id": "1", "name": "test"}])
        self.assertEqual(response.provider, "test_provider")
        self.assertEqual(response.usage, {"tokens": 100})


class TestBaseProvider(unittest.TestCase):
    """Test base LLM provider functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class TestProvider(LLMProvider):
            def __init__(self, api_key: str, model: str):
                super().__init__(api_key, model)

            @property
            def provider_name(self) -> str:
                return "Test"
            
            @property
            def supports_thinking(self) -> bool:
                return True

            def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return super().format_tools_for_llm(tools)
            
            def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return messages

            async def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
                return LLMResponse(content="test response")
        
        self.provider = TestProvider("test_key", "test_model")
    
    def test_initialization(self):
        """Test provider initialization."""
        self.assertEqual(self.provider.api_key, "test_key")
        self.assertEqual(self.provider.model, "test_model")
        self.assertEqual(self.provider.provider_name, "Test")
    
    def test_format_tools_for_llm_base(self):
        """Test base format_tools_for_llm method."""
        tools = [{"name": "test_tool", "description": "Test tool", "inputSchema": {"type": "object"}}]
        result = self.provider.format_tools_for_llm(tools)
        self.assertEqual(result[0]['input_schema'], {"type": "object"})
        
    def test_format_tool_calls_for_execution_base(self):
        """Test base format_tool_calls_for_execution method."""
        tool_calls = [
            {
                "id": "call_123",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"arg1": "value1"}'
                }
            }
        ]
        
        result = self.provider.format_tool_calls_for_execution(tool_calls)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "call_123")
        self.assertEqual(result[0]["name"], "test_tool")
        self.assertEqual(result[0]["input"], {"arg1": "value1"})
    
    def test_format_tool_results_for_conversation_base(self):
        """Test base format_tool_results_for_conversation method."""
        tool_calls = [{"id": "call_1", "name": "test_tool"}]
        tool_outputs = [[{"type": "text", "text": "Result text"}]]
        result, image_parts = self.provider.format_tool_results_for_conversation(tool_calls, tool_outputs)
        self.assertEqual(result[0]['content'], 'Result text')
        self.assertEqual(len(image_parts), 0)


class TestFactory(unittest.TestCase):
    """Test LLM provider factory."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_create_claude_provider(self):
        """Test creating Claude provider."""
        os.environ['ANTHROPIC_API_KEY'] = 'test_key'
        
        provider = create_llm_provider('claude-3-7-sonnet-latest')
        
        self.assertIsInstance(provider, ClaudeProvider)
        self.assertEqual(provider.provider_name, "Claude")
        self.assertEqual(provider.api_key, 'test_key')
        self.assertEqual(provider.model, 'claude-3-7-sonnet-latest')
    
    def test_create_gemini_provider(self):
        """Test creating Gemini provider."""
        os.environ['GEMINI_API_KEY'] = 'test_key'
        
        provider = create_llm_provider('gemini-2.5-flash')
        
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.provider_name, "Gemini")
        self.assertEqual(provider.api_key, 'test_key')
        self.assertEqual(provider.model, 'gemini-2.5-flash')
    
    def test_create_provider_with_override_key(self):
        """Test creating provider with API key override."""
        provider = create_llm_provider('claude-3-7-sonnet-latest', api_key='override_key')
        
        self.assertIsInstance(provider, ClaudeProvider)
        self.assertEqual(provider.api_key, 'override_key')
    
    def test_create_provider_missing_key(self):
        """Test creating provider with missing API key."""
        # Clear the environment variable that was set by the test runner
        if 'ANTHROPIC_API_KEY' in os.environ:
            del os.environ['ANTHROPIC_API_KEY']
            
        with self.assertRaises(ValueError) as context:
            create_llm_provider('claude-3-7-sonnet-latest')
        
        self.assertIn("ANTHROPIC_API_KEY not found", str(context.exception))
    
    def test_create_provider_unsupported_model(self):
        """Test creating provider with unsupported model."""
        with self.assertRaises(ValueError) as context:
            create_llm_provider('unsupported-model')
        
        self.assertIn("Unsupported model", str(context.exception))


class TestClaudeProvider(unittest.TestCase):
    """Test Claude provider functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('llm_providers.claude_provider.anthropic.Anthropic'):
            self.provider = ClaudeProvider('test_key', 'claude-3-7-sonnet-latest')
    
    def test_initialization(self):
        """Test Claude provider initialization."""
        self.assertEqual(self.provider.provider_name, "Claude")
        self.assertTrue(self.provider.supports_thinking)
        self.assertEqual(self.provider.api_key, 'test_key')
        self.assertEqual(self.provider.model, 'claude-3-7-sonnet-latest')
    
    def test_format_tools(self):
        """Test Claude tools formatting."""
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        
        result = self.provider.format_tools(tools)
        expected = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        self.assertEqual(result, expected)
    
    def test_format_messages_basic(self):
        """Test Claude message formatting."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"}
        ]
        
        result = self.provider.format_messages(messages)
        
        # System message should be excluded
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "assistant")
    
    def test_format_messages_with_tool_calls(self):
        """Test Claude message formatting with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll use a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"param": "value"}'
                        }
                    }
                ]
            }
        ]
        
        result = self.provider.format_messages(messages)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "assistant")
        self.assertEqual(len(result[0]["content"]), 2)  # text + tool_use
        self.assertEqual(result[0]["content"][0]["type"], "text")
        self.assertEqual(result[0]["content"][1]["type"], "tool_use")
    
    def test_format_messages_tool_results(self):
        """Test Claude message formatting with tool results."""
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "Result"}
                ]
            }
        ]
        
        result = self.provider.format_messages(messages)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
    
    def test_extract_system_message(self):
        """Test system message extraction."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]
        
        result = self.provider._extract_system_message(messages)
        self.assertEqual(result, "System message")
    
    def test_extract_system_message_none(self):
        """Test system message extraction when none exists."""
        messages = [{"role": "user", "content": "User message"}]
        
        result = self.provider._extract_system_message(messages)
        self.assertIsNone(result)


class TestGeminiProvider(unittest.TestCase):
    """Test Gemini provider functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('llm_providers.gemini_provider.genai.Client'):
            self.provider = GeminiProvider('test_key', 'gemini-2.5-flash')
    
    def test_initialization(self):
        """Test Gemini provider initialization."""
        self.assertEqual(self.provider.provider_name, "Gemini")
        self.assertTrue(self.provider.supports_thinking)  # 2.5 model
        self.assertEqual(self.provider.api_key, 'test_key')
        self.assertEqual(self.provider.model, 'gemini-2.5-flash')
    
    def test_supports_thinking_false(self):
        """Test thinking support for non-2.5 models."""
        with patch('llm_providers.gemini_provider.genai.Client'):
            provider = GeminiProvider('test_key', 'gemini-1.5-pro')
            self.assertFalse(provider.supports_thinking)
    
    def test_format_tools(self):
        """Test Gemini tools formatting."""
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        
        result = self.provider.format_tools(tools)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].function_declarations), 1)
        self.assertEqual(result[0].function_declarations[0].name, "test_tool")
    
    def test_count_images_in_messages(self):
        """Test image counting in messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Text"},
                    {"type": "image", "source": {"type": "base64", "data": "image1"}},
                    {"type": "image", "source": {"type": "base64", "data": "image2"}}
                ]
            }
        ]
        
        result = self.provider._count_images_in_messages(messages)
        self.assertEqual(result, 2)
    
    def test_format_tool_results_for_conversation(self):
        """Test Gemini tool results formatting."""
        tool_calls = [{"id": "call_1", "name": "test_tool"}]
        tool_outputs = [[
            {"type": "text", "text": '{"status": "success", "data": "test"}'},
            {"type": "image", "source": {"type": "base64", "data": "image_data"}}
        ]]
        
        result, image_parts = self.provider.format_tool_results_for_conversation(tool_calls, tool_outputs)
        
        self.assertEqual(len(result), 3)  # tool_result + text + image
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_name"], "test_tool")
        self.assertIn("status", result[0]["content"])  # JSON should be formatted
        self.assertEqual(len(image_parts), 1)


# ==============================================================================
# OpenAI Provider Tests
# ==============================================================================
class TestOpenAIProvider(unittest.TestCase):
    """Tests for the OpenAIProvider."""

    def setUp(self):
        """Set up for OpenAI provider tests."""
        self.api_key = "test_openai_key"
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.provider = OpenAIProvider(api_key=self.api_key, model="gpt-4o-mini")

    def tearDown(self):
        """Tear down after tests."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
            
    def test_initialization_with_key(self):
        """Test provider initializes correctly with an API key."""
        self.assertEqual(self.provider.api_key, self.api_key)
        self.assertIsInstance(self.provider.client, AsyncOpenAI)

    def test_initialization_from_env(self):
        """Test provider initializes correctly from environment variables."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        self.assertEqual(provider.api_key, self.api_key)

    def test_initialization_missing_key(self):
        """Test provider raises error if API key is missing."""
        del os.environ["OPENAI_API_KEY"]
        with self.assertRaises(ValueError) as context:
            OpenAIProvider(model="gpt-4o-mini")
        self.assertIn("OPENAI_API_KEY not found", str(context.exception))
        
    def test_format_tools(self):
        """Test formatting tools for OpenAI API."""
        tools = [{"name": "tool1", "description": "desc1", "input_schema": {"type": "object"}}]
        formatted = self.provider.format_tools(tools)
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]['type'], 'function')
        self.assertEqual(formatted[0]['function']['name'], 'tool1')
        self.assertEqual(formatted[0]['function']['parameters'], {"type": "object"})

    @patch('llm_providers.openai_provider.AsyncOpenAI')
    def test_generate_response_streaming(self, mock_openai_client):
        """Test generating a streaming response."""
        # Setup mock stream - create returns a coroutine that resolves to an async generator
        async def mock_stream_gen():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello ", tool_calls=None))], usage=None)
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="World", tool_calls=None))], usage=None)

        # Mock the create method to return a coroutine that resolves to the async generator
        async def mock_create(**kwargs):
            return mock_stream_gen()

        mock_openai_client.return_value.chat.completions.create = mock_create

        provider = OpenAIProvider(api_key=self.api_key, model="gpt-4o-mini")
        
        async def run_test():
            response = await provider.generate_response(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                temperature=0.5,
                thinking_enabled=False,
                thinking_budget=0
            )
            self.assertEqual(response.content, "Hello World")
            self.assertIsNone(response.tool_calls)

        asyncio.run(run_test())

    @patch('llm_providers.openai_provider.AsyncOpenAI')
    def test_tool_call_streaming(self, mock_openai_client):
        """Test generating a response with a tool call."""
        
        async def mock_stream_gen():
            # First chunk with tool call details
            mock_function = MagicMock()
            mock_function.name = "test_tool"
            mock_function.arguments = ""
            
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=None, tool_calls=[
                MagicMock(index=0, id="call_123", function=mock_function)
            ]))], usage=None)
            
            # Argument chunks
            mock_function2 = MagicMock()
            mock_function2.name = None
            mock_function2.arguments = '{"arg1":'
            
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=None, tool_calls=[
                MagicMock(index=0, id=None, function=mock_function2)
            ]))], usage=None)
            
            mock_function3 = MagicMock()
            mock_function3.name = None
            mock_function3.arguments = '"value1"}'
            
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=None, tool_calls=[
                MagicMock(index=0, id=None, function=mock_function3)
            ]))], usage=None)
            
            # Final usage chunk
            yield MagicMock(usage=MagicMock(
                dict=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            ), choices=[MagicMock(delta=MagicMock(content=None, tool_calls=None))])

        # Mock the create method to return a coroutine that resolves to the async generator
        async def mock_create(**kwargs):
            return mock_stream_gen()

        mock_openai_client.return_value.chat.completions.create = mock_create
        
        provider = OpenAIProvider(api_key=self.api_key, model="gpt-4o-mini")

        async def run_test():
            response = await provider.generate_response(
                messages=[{"role": "user", "content": "Use a tool"}],
                tools=self.provider.format_tools([{"name": "test_tool", "description": "Test tool", "input_schema": {}}]),
                temperature=0.5,
                thinking_enabled=False,
                thinking_budget=0
            )

            self.assertIsNone(response.content)
            self.assertIsNotNone(response.tool_calls)
            self.assertEqual(len(response.tool_calls), 1)
            
            tool_call = response.tool_calls[0]
            self.assertEqual(tool_call['function']['name'], 'test_tool')
            self.assertEqual(tool_call['function']['arguments'], '{"arg1":"value1"}')
            
            # Test usage parsing
            self.assertEqual(response.usage['input_tokens'], 10)
            self.assertEqual(response.usage['output_tokens'], 20)
            self.assertEqual(response.usage['total_tokens'], 30)

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main() 