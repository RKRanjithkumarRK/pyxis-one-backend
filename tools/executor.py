"""
Tool dispatcher — routes tool_name + input_data to the correct implementation.
All tools are async and return strings.
"""

from __future__ import annotations


async def execute_tool(name: str, input_data: dict) -> str:
    if name == "web_search":
        from tools.web_search import search
        query = input_data.get("query", "")
        num = input_data.get("num_results", 5)
        if not query:
            return "Error: search query is required."
        return await search(query, num_results=int(num))

    elif name == "code_interpreter":
        from tools.code_sandbox import execute
        code = input_data.get("code", "")
        language = input_data.get("language", "python")
        if not code:
            return "Error: code is required."
        return await execute(code, language=language)

    elif name == "read_file":
        # file_id lookup — returns pre-extracted content from FileUpload table
        from tools._file_lookup import lookup_file
        file_id = input_data.get("file_id", "")
        extract_type = input_data.get("extract_type", "full")
        if not file_id:
            return "Error: file_id is required."
        return await lookup_file(file_id, extract_type)

    else:
        return f"Unknown tool: {name}"
