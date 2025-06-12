# SO-ARM100 Robot Control with MCP

[![Watch the full tutorial](https://img.youtube.com/vi/EmpQQd7jRqs/0.jpg)](https://youtu.be/EmpQQd7jRqs)

A companion repository to my video about MCP server for the robot:
- **MCP Server** for LLM-based AI agents (Claude Desktop, Cursor, Windsurf, etc.) to control the robot
- **Direct keyboard control** for manual operation


This repository suppose to work with the SO-ARM100 / 101 robots. Refer to [lerobot SO-101 setup guide](https://huggingface.co/docs/lerobot/so101) for the detailed instructions on how to setup the robot.

If you want to know more about MCP refer to the [official MCP documentation](https://github.com/modelcontextprotocol/python-sdk)


## Quick Start

### 1. Install Dependencies

For simplicity I use simple pip instead of uv that is often recommended in MCP tutorials - it works just fine.

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

It may be required to install lerobot separately, just use the official instructions from the [lerobot repository](https://github.com/huggingface/lerobot)


### 2. Connect Your Robot
- Connect SO-ARM100 via USB
- Update `config.py` with your serial port (e.g., `/dev/tty.usbmodem58FD0168731`)
- Connect cameras (optional but recommended) and update `config.py` with the correct indices

### 3. Use the robot

**🔍 Check Robot Status:**
```bash
python check_positions.py
```

This will show you the current robot state without actual control. Move your robot manually to make sure it is properly calibrated and configured.

**🎮 Manual Keyboard Control:**
```bash
python keyboard_controller.py
```

Now you can try to control the robot manually using the keyboard. Test it before moving on to the MCP step, to make sure it works properly.

**🛠️ MCP server in the dev mode**
```bash
mcp dev mcp_robot_server.py
```

Final test step - to debug the MCP server, use the UI to connect to it and try to send some requests.

**🤖 AI Agent Control (MCP Server):**

WARNING: using MCP server itself is free, but it requires MCP client that will send requests to some LLM. Generally it is not free - and controlling the robot with MCP can become expensive, as it sends multiple agentic requests with images that use a lot of tokens. Make sure you understand and control your token usage and corresponding costs before doing it. The actual cost depends on the client and models you use, and it is your responsibility to monitor and control it.

```bash
mcp run mcp_robot_server.py --transport SELECTED_TRANSPORT
```

Supports: `stdio`, `sse`, `streamable-http`

Now your server can be added to any MCP client.

## Connecting MCP Clients

Different clients can support different transports, you can choose the one that works best for you. The functionality is the same.

### STDIO transport

Add to your MCP configuration:
```json
{
  "mcpServers": {
    "SO-ARM100 robot controller": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp_robot_server.py"]
    }
  }
}
```

### SEE transport

Run the server in terminal with the SSE transport:
```bash
mcp run mcp_robot_server.py --transport sse
```

Add to your MCP configuration:
```json
{
  "mcpServers": {
    "SO-ARM100 robot controller": {
      "url": "http://127.0.0.1:3001/sse"
    }
  }
}
```

### Streamed-HTTP transport

It is suppose to be a replacement for SSE but currently not so many clients support it.

Run the server in terminal with the Streamed-HTTP transport:
```bash
mcp run mcp_robot_server.py --transport streamable-http
```

Add to your MCP configuration:
```json
{
  "mcpServers": {
    "SO-ARM100 robot controller": {
      "url": "http://127.0.0.1:3001/mcp"
    }
  }
}
```

## Using the robot with MCP

Now you can go to you Client and it should be able to control the robot when you give it the natural language instructions.
