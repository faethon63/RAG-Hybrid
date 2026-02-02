"""
File Tools Module
Secure file system operations with path validation.
"""

import os
import re
import fnmatch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class PathSecurityError(Exception):
    """Raised when a path operation violates security constraints."""
    pass


def convert_windows_to_wsl(path: str) -> str:
    """
    Convert a Windows path to WSL path if running on Linux.
    E.g., 'C:\\Users\\name\\Documents' -> '/mnt/c/Users/name/Documents'
    """
    import platform
    if platform.system() != "Linux":
        return path

    # Check if it looks like a Windows path
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"

    # Also handle forward-slash Windows paths like C:/Users/...
    if len(path) >= 2 and path[1] == ':' or (len(path) >= 3 and path[0].isalpha() and path[1:3] == ':/'):
        drive = path[0].lower()
        rest = path[2:] if path[1] == ':' else path[3:]
        rest = rest.replace('\\', '/')
        return f"/mnt/{drive}{rest}"

    return path


class FileTools:
    """
    Secure file operations with path restrictions.

    Operations are only allowed within explicitly allowed paths
    defined in the project configuration.
    """

    @staticmethod
    def validate_path(path: str, allowed_paths: List[str]) -> Path:
        """
        Validate that a path is within allowed directories.

        Args:
            path: The path to validate
            allowed_paths: List of allowed directory paths

        Returns:
            Resolved Path object if valid

        Raises:
            PathSecurityError: If path is outside allowed directories
        """
        if not allowed_paths:
            raise PathSecurityError(
                "No allowed paths configured for this project. "
                "Go to Project Settings and add at least one allowed path."
            )

        # Convert Windows paths to WSL if needed
        converted_path = convert_windows_to_wsl(path)
        logger.debug(f"Path conversion: '{path}' -> '{converted_path}'")

        # Resolve to absolute path
        try:
            resolved = Path(converted_path).resolve()
        except Exception as e:
            raise PathSecurityError(f"Invalid path format '{path}': {e}")

        # Check if path is within any allowed directory
        checked_paths = []
        for allowed in allowed_paths:
            # Also convert allowed paths from Windows to WSL format
            converted_allowed = convert_windows_to_wsl(allowed)
            try:
                allowed_resolved = Path(converted_allowed).resolve()
                checked_paths.append(str(allowed_resolved))
                resolved.relative_to(allowed_resolved)
                return resolved
            except ValueError:
                continue
            except Exception as e:
                logger.warning(f"Error resolving allowed path '{allowed}': {e}")
                continue

        # Build helpful error message
        raise PathSecurityError(
            f"Path '{path}' (resolved: {resolved}) is outside allowed directories.\n"
            f"Allowed paths: {checked_paths}\n"
            f"Tip: Update project settings to add the directory you want to access."
        )

    @staticmethod
    def read_file(path: str, allowed_paths: List[str]) -> Dict[str, Any]:
        """
        Read file contents.

        Args:
            path: Path to the file
            allowed_paths: List of allowed directory paths

        Returns:
            Dict with 'success', 'content' or 'error'
        """
        try:
            validated_path = FileTools.validate_path(path, allowed_paths)

            if not validated_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            if not validated_path.is_file():
                return {"success": False, "error": f"Not a file: {path}"}

            # Size limit: 1MB
            if validated_path.stat().st_size > 1024 * 1024:
                return {"success": False, "error": "File too large (>1MB)"}

            content = validated_path.read_text(encoding="utf-8", errors="replace")
            return {
                "success": True,
                "content": content,
                "path": str(validated_path),
                "size": len(content),
            }

        except PathSecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return {"success": False, "error": f"Failed to read file: {e}"}

    @staticmethod
    def write_file(path: str, content: str, allowed_paths: List[str]) -> Dict[str, Any]:
        """
        Write content to a file.

        Args:
            path: Path to the file
            content: Content to write
            allowed_paths: List of allowed directory paths

        Returns:
            Dict with 'success' and details or 'error'
        """
        try:
            validated_path = FileTools.validate_path(path, allowed_paths)

            # Create parent directories if needed
            validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            validated_path.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "path": str(validated_path),
                "size": len(content),
                "message": f"Successfully wrote {len(content)} characters to {path}",
            }

        except PathSecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return {"success": False, "error": f"Failed to write file: {e}"}

    @staticmethod
    def list_dir(path: str, allowed_paths: List[str]) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            path: Path to the directory
            allowed_paths: List of allowed directory paths

        Returns:
            Dict with 'success', 'entries' or 'error'
        """
        try:
            validated_path = FileTools.validate_path(path, allowed_paths)

            if not validated_path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}

            if not validated_path.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            entries = []
            for entry in sorted(validated_path.iterdir()):
                entry_info = {
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "path": str(entry),
                }
                if entry.is_file():
                    try:
                        entry_info["size"] = entry.stat().st_size
                    except Exception:
                        pass
                entries.append(entry_info)

            return {
                "success": True,
                "path": str(validated_path),
                "entries": entries,
                "count": len(entries),
            }

        except PathSecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return {"success": False, "error": f"Failed to list directory: {e}"}

    @staticmethod
    def search_files(
        path: str,
        pattern: str,
        allowed_paths: List[str],
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Search for files matching a pattern.

        Args:
            path: Base directory to search in
            pattern: Glob pattern (e.g., "*.py", "**/*.txt")
            allowed_paths: List of allowed directory paths
            max_results: Maximum number of results to return

        Returns:
            Dict with 'success', 'matches' or 'error'
        """
        try:
            validated_path = FileTools.validate_path(path, allowed_paths)

            if not validated_path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}

            if not validated_path.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            matches = []
            for match in validated_path.glob(pattern):
                if len(matches) >= max_results:
                    break
                matches.append({
                    "name": match.name,
                    "path": str(match),
                    "type": "directory" if match.is_dir() else "file",
                })

            return {
                "success": True,
                "base_path": str(validated_path),
                "pattern": pattern,
                "matches": matches,
                "count": len(matches),
                "truncated": len(matches) >= max_results,
            }

        except PathSecurityError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error searching files in {path}: {e}")
            return {"success": False, "error": f"Failed to search files: {e}"}


# Import PDF tools
from pdf_tools import PDF_TOOL_DEFINITIONS, execute_pdf_tool

# Tool definitions for LLM prompting
TOOL_DEFINITIONS = """
Available file tools:

1. read_file - Read the contents of a file
   Usage: <tool>read_file</tool><path>FILE_PATH</path>

2. write_file - Write content to a file
   Usage: <tool>write_file</tool><path>FILE_PATH</path><content>CONTENT_TO_WRITE</content>

3. list_dir - List the contents of a directory
   Usage: <tool>list_dir</tool><path>DIRECTORY_PATH</path>

4. search_files - Search for files matching a pattern
   Usage: <tool>search_files</tool><path>DIRECTORY_PATH</path><pattern>GLOB_PATTERN</pattern>

""" + PDF_TOOL_DEFINITIONS + """
Only use these tools when the user explicitly asks to read, write, or browse files.
Always confirm the file path before writing.
"""


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a tool call from LLM output.

    Expected format:
    <tool>tool_name</tool><path>path_value</path>[<content>...</content>][<pattern>...</pattern>]

    For PDF tools:
    <tool>fill_pdf_form</tool><input>...</input><output>...</output><fields>{...}</fields>

    Returns:
        Dict with 'tool' and parameters, or None if no tool call found
    """
    # Find tool name
    tool_match = re.search(r"<tool>(\w+)</tool>", text)
    if not tool_match:
        return None

    tool_name = tool_match.group(1)

    result = {"tool": tool_name}

    # Find path (for file tools)
    path_match = re.search(r"<path>(.*?)</path>", text, re.DOTALL)
    if path_match:
        result["path"] = path_match.group(1).strip()

    # Find input/output (for PDF tools)
    input_match = re.search(r"<input>(.*?)</input>", text, re.DOTALL)
    if input_match:
        result["input"] = input_match.group(1).strip()

    output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    if output_match:
        result["output"] = output_match.group(1).strip()

    # Find optional content (for write_file)
    content_match = re.search(r"<content>(.*?)</content>", text, re.DOTALL)
    if content_match:
        result["content"] = content_match.group(1)

    # Find optional pattern (for search_files)
    pattern_match = re.search(r"<pattern>(.*?)</pattern>", text, re.DOTALL)
    if pattern_match:
        result["pattern"] = pattern_match.group(1).strip()

    # Find fields (for fill_pdf_form)
    fields_match = re.search(r"<fields>(.*?)</fields>", text, re.DOTALL)
    if fields_match:
        result["fields"] = fields_match.group(1).strip()

    # Find positions (for add_pdf_text)
    positions_match = re.search(r"<positions>(.*?)</positions>", text, re.DOTALL)
    if positions_match:
        result["positions"] = positions_match.group(1).strip()

    # Find form_id and output_dir (for download_form)
    form_id_match = re.search(r"<form_id>(.*?)</form_id>", text, re.DOTALL)
    if form_id_match:
        result["form_id"] = form_id_match.group(1).strip()

    output_dir_match = re.search(r"<output_dir>(.*?)</output_dir>", text, re.DOTALL)
    if output_dir_match:
        result["output_dir"] = output_dir_match.group(1).strip()

    return result


def execute_tool(
    tool_call: Dict[str, Any],
    allowed_paths: List[str]
) -> Dict[str, Any]:
    """
    Execute a parsed tool call.

    Args:
        tool_call: Dict with 'tool' name and parameters
        allowed_paths: List of allowed directory paths

    Returns:
        Result dict from the tool execution
    """
    tool_name = tool_call.get("tool")
    path = tool_call.get("path")

    if not tool_name or not path:
        return {"success": False, "error": "Invalid tool call: missing tool name or path"}

    if tool_name == "read_file":
        return FileTools.read_file(path, allowed_paths)

    elif tool_name == "write_file":
        content = tool_call.get("content")
        if content is None:
            return {"success": False, "error": "write_file requires content"}
        return FileTools.write_file(path, content, allowed_paths)

    elif tool_name == "list_dir":
        return FileTools.list_dir(path, allowed_paths)

    elif tool_name == "search_files":
        pattern = tool_call.get("pattern", "*")
        return FileTools.search_files(path, pattern, allowed_paths)

    # PDF tools
    elif tool_name in ["read_pdf_fields", "fill_pdf_form", "add_pdf_text", "download_form", "verify_pdf"]:
        return execute_pdf_tool(tool_call, allowed_paths)

    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
