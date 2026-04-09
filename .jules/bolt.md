## 2024-04-09 - [Lift Repeated Regex Compilation]
**Learning:** Found multiple instances where regular expressions were compiled dynamically within heavily executed helper functions (e.g. `_safe_json_loads`, `_extract_multi_file_blocks`) using `re.compile()`, causing unnecessary recompilation overhead.
**Action:** Always inspect helper utilities processing parsed LLM outputs or text generation for repeated `re.compile()` calls and lift them to module-level constants.
