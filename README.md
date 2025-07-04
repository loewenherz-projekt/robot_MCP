# SO-ARM100 Robot Control with MCP

[![Watch the full tutorial](https://img.youtube.com/vi/EmpQQd7jRqs/0.jpg)](https://youtu.be/EmpQQd7jRqs)

A companion repository to my video about MCP server for the robot:
- **MCP Server** for LLM-based AI agents (Claude Desktop, Cursor, Windsurf, etc.) to control the robot
- **Direct keyboard control** for manual operation

If you want to know more about MCP refer to the [official MCP documentation](https://github.com/modelcontextprotocol/python-sdk)

This repository suppose to work with the SO-ARM100 / 101 robots. Refer to [lerobot SO-101 setup guide](https://huggingface.co/docs/lerobot/so101) for the detailed instructions on how to setup the robot.

Update! Now it partially supports [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) (only arm, the mobile base control through MCP is TBD).
I also added a simple agent that uses MCP server to control the robot with Claude. It is much more token efficient than many other agents as it is created specifically for this use case. Currently only works with Claude (other LLMs TBD).

After I released the video and this repository, LeRobot released a significant update of the library that breaks the compatibility with the original code.

If you want to use the original code and exactly follow the video, please use [this release](https://github.com/IliaLarchenko/robot_MCP/tree/v0.0.1).

The current version of the code works with the latest LeRobot version. It supports both SO-ARM100 and LeKiwi (arm only, not mobile base yet). I also did a very big refactoring of the code comparing to the original version discussed in the video.

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
- Update `config.py` with your serial port for so-arm (e.g., `/dev/tty.usbmodem58FD0168731`) or robot_ip for lekiwi (e.g., `192.168.1.1`)
- Connect cameras and update `config.py` with the correct indices and names (for `lekiwi` only names are important)

### 3. Use the robot

**🔍 Check Robot Status and Calibration:**
```bash
python check_positions.py
```

This will show you the current robot state without actual control. Move your robot manually to make sure it is properly calibrated and configured.

After the latest update, lerobot is using the normalized joints states instead of degrees. You can update `MOTOR_NORMALIZED_TO_DEGREE_MAPPING` in `config.py` to match your robot calibration. You will need to update these values every time you recalibrate the robot.

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


## Using the Agent

Start the MCP server with the SSE transport

```bash
mcp run mcp_robot_server.py --transport sse
```

Now you can use the agent to control the robot.

```bash
python agent.py
```

You anthropic API key should either be set in the environment variable `ANTHROPIC_API_KEY` or passed as an argument to the agent.py script.

```bash
python agent.py --api-key <your-anthropic-api-key>
```

You can also specify the model to use and whether to show images received from the robot.

```bash
python agent.py --model <your-model> --show-images
```

Other arguments:
--mcp-server-ip - MCP server address, default is 127.0.0.1
--mcp-port - MCP server port, default is 3001
--thinking-budget - Claude thinking budget in tokens, default is 1024 (it is minimum). Higher leads to better reasoning but it is slower and more expensive.
--thinking-every-n - Use thinking every n steps, default is 3.


```bash
python agent.py --mcp-server-ip <your-mcp-host> --mcp-port <your-mcp-port> --thinking-budget <your-thinking-budget> --thinking-every-n <your-thinking-every-n>
```

0 budget will disable thinking, it is the fastest and cheapest option but the success rate will drop a lot in this case. It will mostly work for direct simple instructions but can struggle with complex tasks.
