import re

def is_safe_sql(query: str) -> bool:
    """
    First-principles security: We strictly allow SELECT statements and 
    block any SQL keyword that can mutate data or schema.
    """
    if not query.strip().lower().startswith("select"):
        return False
        
    # Extract only words to prevent blocking column names that contain substrings
    tokens = set(re.findall(r'\b\w+\b', query.lower()))
    
    forbidden_commands = {
        "insert", "update", "delete", "drop", "alter", 
        "create", "truncate", "grant", "revoke", "replace", "commit"
    }
    
    if tokens.intersection(forbidden_commands):
        return False
        
    return True