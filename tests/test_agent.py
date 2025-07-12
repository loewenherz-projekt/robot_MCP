"""
Unit tests for AI agent.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import os
import sys
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import AIAgent
from llm_providers.base_provider import LLMResponse


class TestAIAgent(unittest.TestCase):
    """Test AI agent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        os.environ['ANTHROPIC_API_KEY'] = 'test_key'
        
        # Mock LLM provider
        self.mock_provider = Mock()
        self.mock_provider.provider_name = "TestProvider"
        self.mock_provider.format_tools_for_llm.return_value = []
        self.mock_provider.format_tool_calls_for_execution.return_value = []
        self.mock_provider.format_tool_results_for_conversation.return_value = ([], [])
        
        # Mock MCP session
        self.mock_session = Mock()
        self.mock_session.call_tool = AsyncMock()
        
        # Mock image viewer
        self.mock_image_viewer = Mock()
    
    @patch('agent.create_llm_provider')
    @patch('agent.ImageViewer')
    def test_initialization_default(self, mock_image_viewer_class, mock_create_provider):
        """Test agent initialization with default parameters."""
        mock_create_provider.return_value = self.mock_provider
        mock_image_viewer_class.return_value = self.mock_image_viewer
        
        agent = AIAgent()
        
        self.assertEqual(agent.model, "claude-3-7-sonnet-latest")
        self.assertEqual(agent.mcp_url, "http://127.0.0.1:3001/sse")
        self.assertEqual(agent.thinking_budget, 1024)
        self.assertEqual(agent.thinking_every_n, 1)
        self.assertEqual(agent.conversation_history, [])
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.session)
        self.assertEqual(agent.llm_provider, self.mock_provider)
    
    @patch('agent.create_llm_provider')
    @patch('agent.ImageViewer')
    def test_initialization_custom_params(self, mock_image_viewer_class, mock_create_provider):
        """Test agent initialization with custom parameters."""
        mock_create_provider.return_value = self.mock_provider
        mock_image_viewer_class.return_value = self.mock_image_viewer
        
        agent = AIAgent(
            model="gemini-2.5-flash",
            show_images=True,
            mcp_server_ip="127.0.0.1",
            mcp_port=3002,
            thinking_budget=2048,
            thinking_every_n=5,
            api_key="custom_key"
        )
        
        self.assertEqual(agent.model, "gemini-2.5-flash")
        self.assertEqual(agent.mcp_url, "http://127.0.0.1:3002/sse")
        self.assertEqual(agent.thinking_budget, 2048)
        self.assertEqual(agent.thinking_every_n, 5)
        self.assertTrue(agent.show_images)
        mock_create_provider.assert_called_with("gemini-2.5-flash", "custom_key")
    
    @patch('agent.create_llm_provider')
    def test_initialization_no_image_viewer(self, mock_create_provider):
        """Test agent initialization without image viewer available."""
        mock_create_provider.return_value = self.mock_provider
        
        with patch('agent.IMAGE_VIEWER_AVAILABLE', False):
            agent = AIAgent(show_images=True)
            
            self.assertFalse(agent.show_images)
            self.assertIsNone(agent.image_viewer)
    
    @patch('agent.create_llm_provider')
    def test_execute_mcp_tool_success(self, mock_create_provider):
        """Test successful MCP tool execution."""
        mock_create_provider.return_value = self.mock_provider
        agent = AIAgent()
        agent.session = self.mock_session
        
        # Mock tool result - text item should have 'text' attribute, image should have 'data' and 'mimeType'
        mock_text_item = Mock()
        mock_text_item.text = "Tool executed successfully"
        # Remove data and mimeType from text item
        del mock_text_item.data
        del mock_text_item.mimeType
        
        mock_image_item = Mock()
        mock_image_item.data = "image_data"
        mock_image_item.mimeType = "image/jpeg"
        # Remove text from image item
        del mock_image_item.text
        
        mock_result = Mock()
        mock_result.content = [mock_text_item, mock_image_item]
        self.mock_session.call_tool.return_value = mock_result
        
        # Run async test
        async def run_test():
            result = await agent.execute_mcp_tool("test_tool", {"param": "value"})
            
            self.assertEqual(len(result), 2)  # text + image
            self.assertEqual(result[0]["type"], "text")
            self.assertEqual(result[1]["type"], "image")
            self.mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"})
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_execute_mcp_tool_no_session(self, mock_create_provider):
        """Test MCP tool execution without session."""
        mock_create_provider.return_value = self.mock_provider
        agent = AIAgent()
        agent.session = None
        
        async def run_test():
            result = await agent.execute_mcp_tool("test_tool", {"param": "value"})
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["type"], "text")
            self.assertIn("Error: Not connected", result[0]["text"])
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_execute_mcp_tool_exception(self, mock_create_provider):
        """Test MCP tool execution with exception."""
        mock_create_provider.return_value = self.mock_provider
        agent = AIAgent()
        agent.session = self.mock_session
        
        self.mock_session.call_tool.side_effect = Exception("Tool error")
        
        async def run_test():
            result = await agent.execute_mcp_tool("test_tool", {"param": "value"})
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["type"], "text")
            self.assertIn("Error: Tool error", result[0]["text"])
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_filter_images_from_conversation(self, mock_create_provider):
        """Test image filtering from conversation."""
        mock_create_provider.return_value = self.mock_provider
        agent = AIAgent()
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "source": {"type": "base64", "data": "image_data"}}
                ]
            },
            {
                "role": "assistant",
                "content": "Response"
            }
        ]
        
        filtered = agent._filter_images_from_conversation(conversation)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["role"], "user")
        self.assertEqual(filtered[0]["content"], "Hello")  # Only text should remain
        self.assertEqual(filtered[1]["role"], "assistant")
        self.assertEqual(filtered[1]["content"], "Response")
    
    @patch('agent.create_llm_provider')
    def test_filter_images_preserves_non_image_content(self, mock_create_provider):
        """Test that image filtering preserves non-image content."""
        mock_create_provider.return_value = self.mock_provider
        agent = AIAgent()
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First text"},
                    {"type": "text", "text": "Second text"},
                    {"type": "image", "source": {"type": "base64", "data": "image_data"}}
                ]
            }
        ]
        
        filtered = agent._filter_images_from_conversation(conversation)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered[0]["content"]), 2)  # Two text items should remain
        self.assertEqual(filtered[0]["content"][0]["text"], "First text")
        self.assertEqual(filtered[0]["content"][1]["text"], "Second text")
    
    @patch('agent.create_llm_provider')
    def test_process_with_llm_no_tool_calls(self, mock_create_provider):
        """Test LLM processing without tool calls."""
        mock_create_provider.return_value = self.mock_provider
        
        # Mock LLM response without tool calls
        mock_response = LLMResponse(
            content="Simple response",
            tool_calls=[],
            provider="TestProvider",
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        )
        self.mock_provider.generate_response = AsyncMock(return_value=mock_response)
        
        agent = AIAgent()
        
        async def run_test():
            result = await agent.process_with_llm("Hello")
            
            self.assertEqual(result, "Simple response")
            self.assertEqual(len(agent.conversation_history), 2)  # user + assistant
            self.assertEqual(agent.conversation_history[0]["role"], "user")
            self.assertEqual(agent.conversation_history[1]["role"], "assistant")
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_process_with_llm_with_tool_calls(self, mock_create_provider):
        """Test LLM processing with tool calls."""
        mock_create_provider.return_value = self.mock_provider
        
        # Mock LLM response with tool calls
        mock_response = LLMResponse(
            content="Using tool",
            tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}],
            provider="TestProvider"
        )
        
        # Mock final response after tool execution
        final_response = LLMResponse(
            content="Tool completed",
            tool_calls=[],
            provider="TestProvider"
        )
        
        self.mock_provider.generate_response = AsyncMock(side_effect=[mock_response, final_response])
        self.mock_provider.format_tool_calls_for_execution.return_value = [
            {"id": "call_1", "name": "test_tool", "input": {}}
        ]
        self.mock_provider.format_tool_results_for_conversation.return_value = (
            [{"type": "tool_result", "tool_use_id": "call_1", "content": "Success"}],
            []
        )
        
        agent = AIAgent()
        agent.session = self.mock_session
        
        # Mock tool execution
        mock_result = Mock()
        mock_result.content = [Mock(text="Tool result")]
        self.mock_session.call_tool.return_value = mock_result
        
        async def run_test():
            result = await agent.process_with_llm("Use a tool")
            
            self.assertEqual(result, "Tool completed")
            # Should have: user message, assistant with tool call, tool result, final assistant
            self.assertGreater(len(agent.conversation_history), 2)
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_process_with_llm_thinking_enabled(self, mock_create_provider):
        """Test LLM processing with thinking enabled."""
        mock_create_provider.return_value = self.mock_provider
        
        mock_response = LLMResponse(
            content="Response with thinking",
            thinking="I need to think about this",
            tool_calls=[],
            provider="TestProvider"
        )
        self.mock_provider.generate_response = AsyncMock(return_value=mock_response)
        
        agent = AIAgent(thinking_budget=1024, thinking_every_n=1)
        
        async def run_test():
            result = await agent.process_with_llm("Complex question")
            
            self.assertEqual(result, "Response with thinking")
            # Verify thinking was enabled in the call
            call_args = self.mock_provider.generate_response.call_args
            self.assertTrue(call_args.kwargs['thinking_enabled'])
            self.assertEqual(call_args.kwargs['thinking_budget'], 1024)
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_process_with_llm_exception_handling(self, mock_create_provider):
        """Test LLM processing exception handling."""
        mock_create_provider.return_value = self.mock_provider
        
        self.mock_provider.generate_response = AsyncMock(side_effect=Exception("LLM error"))
        
        agent = AIAgent()
        
        async def run_test():
            result = await agent.process_with_llm("Test input")
            
            self.assertIn("An error occurred: LLM error", result)
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_cleanup(self, mock_create_provider):
        """Test agent cleanup."""
        mock_create_provider.return_value = self.mock_provider
        
        agent = AIAgent()
        agent.image_viewer = self.mock_image_viewer
        
        agent.cleanup()
        
        self.mock_image_viewer.cleanup.assert_called_once()
    
    @patch('agent.create_llm_provider')
    def test_cleanup_no_image_viewer(self, mock_create_provider):
        """Test agent cleanup without image viewer."""
        mock_create_provider.return_value = self.mock_provider
        
        agent = AIAgent()
        agent.image_viewer = None
        
        # Should not raise exception
        agent.cleanup()
    
    @patch('agent.create_llm_provider')
    @patch('agent.sse_client')
    @patch('agent.ClientSession')
    def test_run_cli_success(self, mock_client_session, mock_sse_client, mock_create_provider):
        """Test CLI run success flow."""
        mock_create_provider.return_value = self.mock_provider
        
        # Mock MCP session
        mock_session_instance = Mock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.list_tools = AsyncMock()
        mock_session_instance.list_tools.return_value.tools = [
            Mock(model_dump=Mock(return_value={"name": "test_tool", "description": "Test"}))
        ]
        mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_client_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock SSE client
        mock_read, mock_write = Mock(), Mock()
        mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)
        
        agent = AIAgent()
        
        # Mock input to quit immediately
        with patch('builtins.input', side_effect=['quit']):
            async def run_test():
                await agent.run_cli()
                
                mock_session_instance.initialize.assert_called_once()
                mock_session_instance.list_tools.assert_called_once()
            
            asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    @patch('agent.sse_client')
    def test_run_cli_connection_error(self, mock_sse_client, mock_create_provider):
        """Test CLI run with connection error."""
        mock_create_provider.return_value = self.mock_provider
        
        mock_sse_client.side_effect = Exception("Connection failed")
        
        agent = AIAgent()
        
        async def run_test():
            await agent.run_cli()
            # Should handle exception gracefully
        
        asyncio.run(run_test())


class TestAgentUtilities(unittest.TestCase):
    """Test agent utility functions and edge cases."""
    
    @patch('agent.create_llm_provider')
    def test_temperature_logic(self, mock_create_provider):
        """Test temperature logic for thinking vs non-thinking."""
        mock_provider = Mock()
        mock_provider.provider_name = "TestProvider"
        mock_provider.generate_response = AsyncMock(return_value=LLMResponse(content="test"))
        mock_create_provider.return_value = mock_provider
        
        agent = AIAgent(thinking_budget=1024, thinking_every_n=2)
        
        async def run_test():
            # First call - iteration 0 - should use thinking (temperature 1.0)
            await agent.process_with_llm("test1")
            first_call_args = mock_provider.generate_response.call_args
            self.assertEqual(first_call_args.kwargs['temperature'], 1.0)
            
            # Reset call history and create new agent for second test
            mock_provider.reset_mock()
            agent2 = AIAgent(thinking_budget=1024, thinking_every_n=1)  # Every iteration
            
            # Second call - iteration 0 again - should use thinking (temperature 1.0)
            await agent2.process_with_llm("test2")
            second_call_args = mock_provider.generate_response.call_args
            self.assertEqual(second_call_args.kwargs['temperature'], 1.0)
        
        asyncio.run(run_test())
    
    @patch('agent.create_llm_provider')
    def test_max_iterations_limit(self, mock_create_provider):
        """Test maximum iterations limit."""
        mock_provider = Mock()
        mock_provider.provider_name = "TestProvider"
        mock_provider.format_tool_calls_for_execution.return_value = [
            {"id": "call_1", "name": "test_tool", "input": {}}
        ]
        mock_provider.format_tool_results_for_conversation.return_value = ([], [])
        
        # Always return tool calls to trigger iterations
        mock_response = LLMResponse(
            content="Using tool",
            tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}],
            provider="TestProvider"
        )
        mock_provider.generate_response = AsyncMock(return_value=mock_response)
        mock_create_provider.return_value = mock_provider
        
        agent = AIAgent()
        agent.session = Mock()
        agent.session.call_tool = AsyncMock(return_value=Mock(content=[Mock(text="result")]))
        
        async def run_test():
            result = await agent.process_with_llm("Test endless loop")
            
            self.assertIn("Completed 100 iterations", result)
        
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main() 