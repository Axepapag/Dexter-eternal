__tool_prefix__ = "ask_ops"

def request_user_location() -> dict:
    """
    Generates a prompt to ask the user for their current city or location.

    This tool does not perform direct I/O (like input()) but returns the 
    structured message intended for the user interface layer.

    Returns:
        dict: A dictionary with a success status and the prompt data.
              Example: {"success": True, "data": {"prompt": "..."}}
    """
    try:
        prompt_message = "Please provide your current city or location."
        
        return {
            "success": True,
            "data": {
                "prompt": prompt_message,
                "intent": "location_request"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate location request: {str(e)}"
        }