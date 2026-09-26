"""The harness actions exposed as native model function tools."""

ACTION_TOOLS = [
    {"type":"function","function":{"name":"list_files","description":"List repository files.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"read_file","description":"Read a repository file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"search_text","description":"Search repository file contents.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"search_names","description":"Search repository filenames.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"write_file","description":"Create or update a repository file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"run_command","description":"Run a controlled command in the repository.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"git_status","description":"Read repository Git status.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"finish","description":"Finish the task with a summary.","parameters":{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]}}},
]
ACTION_NAMES = {tool["function"]["name"] for tool in ACTION_TOOLS}
