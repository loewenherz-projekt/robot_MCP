#!/usr/bin/env python3
"""
AI Agent with CLI interface that uses Claude and MCP tools.
Connects to MCP servers via SSE transport for autonomous task execution.
"""

import asyncio
import json
import os
import sys
import argparse
from typing import Dict, List, Any

from mcp import ClientSession
from mcp.client.sse import sse_client
import anthropic

try:
    from agent_utils import ImageViewer
    IMAGE_VIEWER_AVAILABLE = True
except ImportError:
    IMAGE_VIEWER_AVAILABLE = False

class AIAgent:
    """AI Agent that uses Claude with MCP tools for autonomous task execution."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514", show_images: bool = False, 
                 mcp_server_ip: str = "127.0.0.1", mcp_port: int = 3001, thinking_budget: int = 1000):
        self.api_key = api_key
        self.model = model
        self.mcp_url = f"http://{mcp_server_ip}:{mcp_port}/sse"
        self.thinking_budget = thinking_budget
        self.conversation_history = []
        self.tools = []
        self.session = None
        self.claude_client = anthropic.Anthropic(api_key=api_key)
        
        # Optional image viewer
        self.show_images = show_images and IMAGE_VIEWER_AVAILABLE
        self.image_viewer = ImageViewer() if self.show_images else None
        
        if show_images and not IMAGE_VIEWER_AVAILABLE:
            print("⚠️  Image display requested but agent_utils.py not available or missing dependencies")

    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute an MCP tool and return formatted content blocks for Claude."""
        if not self.session:
            return [{"type": "text", "text": "Error: Not connected to MCP server"}]

        try:
            result = await self.session.call_tool(tool_name, arguments)
            content_parts = []
            image_count = 0

            if hasattr(result.content, '__iter__') and not isinstance(result.content, (str, bytes)):
                for item in result.content:
                    if hasattr(item, 'data') and hasattr(item, 'mimeType'):
                        image_count += 1
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": item.mimeType,
                                "data": item.data
                            }
                        })
                    elif hasattr(item, 'text'):
                        content_parts.append({"type": "text", "text": item.text})
                    elif isinstance(item, dict):
                        content_parts.append({"type": "text", "text": json.dumps(item, indent=2)})
                    else:
                        content_parts.append({"type": "text", "text": str(item)})
            else:
                content_parts.append({"type": "text", "text": str(result.content)})

            if image_count > 0:
                print(f"🔧 {tool_name}: returned text + {image_count} images")
            else:
                print(f"🔧 {tool_name}: returned text")
            
            return content_parts

        except Exception as e:
            print(f"❌ Error executing {tool_name}: {str(e)}")
            return [{"type": "text", "text": f"Error during tool execution: {str(e)}"}]

    def format_tools_for_claude(self) -> List[Dict[str, Any]]:
        """Format MCP tools for Claude's API."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}})
            }
            for tool in self.tools
        ]

    def _filter_images_from_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove images from conversation to prevent token accumulation."""
        filtered_conversation = []
        for msg in conversation:
            if isinstance(msg.get('content'), list):
                filtered_content = [
                    content for content in msg['content'] 
                    if not (isinstance(content, dict) and content.get('type') == 'image')
                ]
                if filtered_content:
                    filtered_conversation.append({
                        "role": msg["role"],
                        "content": filtered_content
                    })
            else:
                filtered_conversation.append(msg)
        return filtered_conversation

    async def process_with_claude(self, user_input: str) -> str:
        """Process user input with Claude in autonomous mode."""
        system_prompt = """You are an AI assistant with access to tools. 
        Use them as needed to control a robot and complete tasks.
        Move step by step, evaluate the results of you action after each step.
        Make sure that the step is successfully completed before moving to the next step.
        Before side moves make sure you are high enough to avoid collisions.
        IMPORTANT: When analyzing images pay extra attention to the relative position of the object and the robot.
        E.g. right on the image can be left from the robot perspective and vice versa.
        If in doubt think twice.
        """
        
        self.conversation_history.append({"role": "user", "content": user_input})

        max_iterations = 100
        for iteration in range(max_iterations):
            try:
                with self.claude_client.messages.stream(
                    model=self.model,
                    max_tokens=16000,
                    system=system_prompt,
                    messages=self.conversation_history,
                    tools=self.format_tools_for_claude(),
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    },
                ) as stream:
                    thinking_started = False
                    response_started = False
                    response_content = []
                    thinking_content = []
                    usage_info = None

                    for event in stream:
                        if event.type == "message_start":
                            usage_info = event.message.usage
                        elif event.type == "content_block_start":
                            thinking_started = False
                            response_started = False
                        elif event.type == "content_block_delta":
                            if event.delta.type == "thinking_delta":
                                if not thinking_started:
                                    print("🧠 Thinking: ", end="", flush=True)
                                    thinking_started = True
                                print(event.delta.thinking, end="", flush=True)
                                thinking_content.append(event.delta.thinking)
                            elif event.delta.type == "text_delta":
                                if not response_started:
                                    print("\n🤖 Claude: ", end="", flush=True)
                                    response_started = True
                                print(event.delta.text, end="", flush=True)
                                response_content.append(event.delta.text)
                        elif event.type == "content_block_stop":
                            if thinking_started or response_started:
                                print()  # New line after block

                    # Get the final message from the stream
                    response = stream.get_final_message()

                if hasattr(response, 'usage'):
                    usage = response.usage
                    print(f"🤖 Claude: {usage.input_tokens}→{usage.output_tokens} tokens")

                assistant_response_content = [block.model_dump() for block in response.content]
                self.conversation_history.append({"role": "assistant", "content": assistant_response_content})
                
                # Show Claude's textual response
                text_blocks = [block for block in response.content if block.type == 'text']
                if text_blocks:
                    claude_text = "".join(block.text for block in text_blocks)
                    if claude_text.strip():
                        print(f"🤖 Claude: {claude_text}")
                
                tool_calls = [block for block in response.content if block.type == 'tool_use']

                if not tool_calls:
                    final_text = "".join(block.get("text", "") for block in assistant_response_content)
                    return final_text

                # Remove images before processing tools to prevent accumulation
                # It saves a lot of tokens as we don't send old images to the LLM
                self.conversation_history = self._filter_images_from_conversation(self.conversation_history)

                tool_results_with_images = []
                
                for tool_call in tool_calls:
                    tool_output_parts = await self.execute_mcp_tool(tool_call.name, tool_call.input)
                    
                    text_parts = []
                    image_parts = []

                    for part in tool_output_parts:
                        if part['type'] == 'image':
                            image_parts.append(part)
                        else:
                            text_parts.append(part['text'])
                    
                    tool_results_with_images.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": "\n".join(text_parts) if text_parts else "Tool executed successfully."
                    })
                    
                    for i, image_part in enumerate(image_parts, 1):
                        tool_results_with_images.append({
                            "type": "text",
                            "text": f"Image {i}:"
                        })
                        tool_results_with_images.append(image_part)
                
                # Update image viewer with new images
                if image_parts and self.image_viewer:
                    self.image_viewer.update(image_parts)
                
                self.conversation_history.append({"role": "user", "content": tool_results_with_images})

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                return f"An error occurred: {str(e)}"

        return f"Agent completed {max_iterations} iterations."

    def cleanup(self):
        """Clean up resources when shutting down."""
        if self.image_viewer:
            self.image_viewer.cleanup()

    async def run_cli(self):
        """Run the main command-line interface loop."""
        print("\n🤖 AI Agent with Claude & MCP Tools")
        print("=" * 50)
        print("Connecting to MCP server...")

        try:
            async with sse_client(self.mcp_url) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    tools_response = await session.list_tools()
                    self.tools = [tool.model_dump() for tool in tools_response.tools]
                    
                    print("✅ Connected to MCP server")
                    print(f"Available tools: {', '.join(tool['name'] for tool in self.tools)}")
                    print("\nType your instructions or 'quit' to exit.")

                    while True:
                        user_input = input("\n> ").strip()
                        if not user_input:
                            continue
                        if user_input.lower() in ['quit', 'exit']:
                            print("Goodbye!")
                            self.cleanup()
                            break

                        print("🤔 Processing...")
                        response_text = await self.process_with_claude(user_input)
                        print(f"\n🤖 Final: {response_text}")

        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            print(f"Make sure the MCP server is running at {self.mcp_url}")
        finally:
            self.cleanup()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Agent with Claude & MCP")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-opus-4-20250514", help="Claude model to use")
    parser.add_argument("--show-images", action="store_true", help="Enable image display window")
    parser.add_argument("--mcp-server-ip", default="127.0.0.1", help="MCP server IP")
    parser.add_argument("--mcp-port", type=int, default=3001, help="MCP server port")
    parser.add_argument("--thinking-budget", type=int, default=1024, help="Claude thinking budget in tokens")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable or --api-key argument is required.")
        
    agent = AIAgent(api_key, args.model, args.show_images, args.mcp_server_ip, args.mcp_port, args.thinking_budget)
    await agent.run_cli()

if __name__ == "__main__":
    asyncio.run(main())
