from typing import TypedDict, List, Dict

class HomeState(TypedDict):
    user_prompt: str           # "Where is my laptop?"
    target_objects: List[str]  # ["laptop"] (Extracted by Qwen)
    house_memory: Dict         # {"Living Room": {"laptop": [x,y,w,h]}}
    agent_response: str        # The final answer back to the user