"""Entry point for MCP server using repo's own path resolution."""
import sys
import os
from climasafeai.utils.paths import PROJECT_DIR

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from agents.tools.prediction_mcp_tool import run_mcp_server

if __name__ == "__main__":
    stdio = "--stdio" in sys.argv
    run_mcp_server(stdio=stdio)
