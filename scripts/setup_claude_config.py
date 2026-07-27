"""Generate or update ~/.config/Claude/claude_desktop_config.json."""
import json
from pathlib import Path
from climasafeai.utils.paths import PROJECT_DIR

config_dir = Path.home() / ".config" / "Claude"
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "claude_desktop_config.json"

data = json.loads(config_file.read_text()) if config_file.exists() else {}
data.setdefault("mcpServers", {})["climasafeai"] = {
    "command": "uv",
    "args": [
        "run", "--directory", str(PROJECT_DIR),
        "python", "-m", "agents.tools.prediction_mcp_tool", "--stdio",
    ],
}

existing = json.loads(config_file.read_text()) if config_file.exists() else {}
if existing != data:
    config_file.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Config actualizado: {config_file}")
else:
    print("Config ya está actualizado")
