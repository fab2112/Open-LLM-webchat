# Open-LLM-webchat - Process MCPs stdio servers
import os
import json

from typing import List
from mcp import StdioServerParameters  

json_path = "./mcps_config.json"

def load_mcps_stdioserverparameters() -> List[StdioServerParameters]:
    """
    Load MCP server configurations from a JSON file and return a list of server parameters.

    This function reads a JSON configuration file, merges each server's custom environment
    with the system environment, and creates a list of `StdioServerParameters` objects
    representing the configured MCP servers.

    Returns:
        List[StdioServerParameters]: A list of configured MCP server stdio parameters.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    servers = []
    for name, data in config.items():
        # Merge system environment with custom environment if provided
        env = os.environ.copy()
        if "env" in data:
            env.update(data["env"])

        # Create the server parameters
        params = StdioServerParameters(
            command=data["command"],
            args=data.get("args", []),
            env=env
        )
        servers.append(params)

    return servers