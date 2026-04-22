"""
Unified tool schema definitions.
One source of truth — converted to provider-specific format by unified_client.py.
"""

from __future__ import annotations

ALL_TOOLS = {
    "web_search": {
        "name": "web_search",
        "description": (
            "Search the web for current information, recent news, facts, documentation, "
            "or anything that requires up-to-date knowledge beyond your training data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — be specific and use keywords",
                },
                "num_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of results to retrieve (1-10)",
                },
            },
            "required": ["query"],
        },
    },

    "code_interpreter": {
        "name": "code_interpreter",
        "description": (
            "Execute Python code in a secure sandbox. Use for: calculations, data analysis, "
            "generating charts, running algorithms, testing code snippets, file processing. "
            "stdout and any generated images are returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "language": {
                    "type": "string",
                    "enum": ["python"],
                    "default": "python",
                },
            },
            "required": ["code"],
        },
    },

    "read_file": {
        "name": "read_file",
        "description": "Read and analyze the content of an uploaded file by its file_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The file ID returned when the file was uploaded",
                },
                "extract_type": {
                    "type": "string",
                    "enum": ["full", "summary", "tables", "code"],
                    "default": "full",
                },
            },
            "required": ["file_id"],
        },
    },
}


def get_tool_schemas(tool_names: list[str]) -> list[dict]:
    """Return schemas for the given tool names."""
    return [ALL_TOOLS[name] for name in tool_names if name in ALL_TOOLS]


def get_all_schemas() -> list[dict]:
    return list(ALL_TOOLS.values())
