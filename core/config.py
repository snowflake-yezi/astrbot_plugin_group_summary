import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = PACKAGE_ROOT / "fonts"
PLUGIN_MARKER = "group_summary"
MAX_PAGE_SIZE = 1000
MAX_SCAN_PAGES = 20
MAX_STORED_MESSAGES = 20_000
MAX_MESSAGE_LENGTH = 2_000
GROUP_COMMAND_PATTERN = re.compile(r"^/?群聊总结(?:\s+(.+?))?\s*$")
PERSONAL_COMMAND_PATTERN = re.compile(r"^/?个人聊天(记录|详情)(?:\s+(.*))?\s*$")
