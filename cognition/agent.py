import json
import ollama
import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from config import TRACKABLE_OBJECTS, MEMORY_FILE, GEMINI_API_KEY

# Configure the Gemini API once at the top of the file
genai.configure(api_key=GEMINI_API_KEY)

class HomeState(TypedDict):
    user_prompt: str           
    target_objects: List[str]  
    retrieved_memory: str        
    agent_response: str        

def parse_intent(state: HomeState):
    """Brain Node: Uses Qwen to pick targets strictly from the config vocabulary."""
    print("\n[Brain] Parsing intent against config vocabulary...")
    
    prompt = f"""
    You are an intent extractor. User: "{state['user_prompt']}"
    Allowed objects: {json.dumps(TRACKABLE_OBJECTS)}
    
    RULES:
    1. If the user asks for specific objects, return a JSON list of those objects.
    2. If the user asks a general question (e.g., "What do you see?", "Scan the room"), return EXACTLY: ["ALL"]
    3. Return ONLY a valid JSON list.
    """
    
    try:
        response = ollama.chat(model='qwen2.5:1.5b', messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content'].strip()
        content = content.replace('```json', '').replace('```', '').strip()
        
        target_list = json.loads(content)
        
        if "ALL" in target_list:
            validated_targets = ["ALL"]
        else:
            # Strictly validate against config
            validated_targets = [item for item in target_list if item in TRACKABLE_OBJECTS]
            if not validated_targets: 
                validated_targets = ["ALL"]
                
        state['target_objects'] = validated_targets
        print(f"[Brain] Extracted targets: {state['target_objects']}")
        
    except Exception as e:
        state['target_objects'] = ["ALL"]
        
    return state

def fetch_memory(state: HomeState):
    """Brain Node: Aggregates memory by room using pure Python logic."""
    print("[Brain] Fetching from persistent memory...")
    
    if not os.path.exists(MEMORY_FILE):
        state['retrieved_memory'] = "The memory bank is empty."
        return state
        
    with open(MEMORY_FILE, "r") as f:
        memory_bank = json.load(f)
        
    targets = [t.lower() for t in state['target_objects']]
    found_items = set()
    room_summaries = {}
    
    # 1. Group items by room
    for room_name, room_data in memory_bank.items():
        objects = room_data.get("objects", {})
        for obj_name, data in objects.items():
            clean_obj_name = obj_name.lower()
            
            if "all" in targets or clean_obj_name in targets:
                if room_name not in room_summaries:
                    room_summaries[room_name] = []
                
                # Format: "bowl (at 14:22:38)"
                room_summaries[room_name].append(f"{clean_obj_name} (at {data['last_seen']})")
                found_items.add(clean_obj_name)
                
    # 2. Calculate missing items
    missing_targets = []
    if "all" not in targets:
        missing_targets = [t for t in targets if t not in found_items]
        
    # 3. Build a strict, human-readable sentence in Python
    memory_lines = []
    for room, items in room_summaries.items():
        # Changed from "In the {room}, I see:" to "Located in the {room}:"
        memory_lines.append(f"Located in the {room}: {', '.join(items)}.")
        
    if missing_targets:
        # Changed from "I could not find:" to "Missing from memory:"
        memory_lines.append(f"Missing from memory: {', '.join(missing_targets)}.")
        
    if not memory_lines:
        state['retrieved_memory'] = "None of the requested items were found in the house."
    else:
        state['retrieved_memory'] = " ".join(memory_lines)
        
    print(f"[Memory Fetched]: \n{state['retrieved_memory']}")
    return state

def generate_response(state: HomeState):
    """Brain Node: Uses Gemini 1.5 Flash to naturally relay the Python facts."""
    
    system_prompt = """
    You are a precise smart home assistant. Read the provided 'Memory Facts' and relay them to the user naturally.
    
    CRITICAL RULES:
    1. DO NOT invent any objects, rooms, or times.
    2. ONLY mention what is explicitly written in the Memory Facts.
    3. DO NOT state, guess, or assume where the *user* is located. You only know the location of the objects.
    4. Keep the response friendly but objective.
    """
    
    user_data = f"""
    User Question: "{state['user_prompt']}"
    
    Memory Facts: 
    {state['retrieved_memory']}
    """
    
    try:
        # Initialize the Gemini model with the strict system instructions
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_prompt.strip()
        )
        
        # Generate the response
        response = model.generate_content(user_data.strip())
        state['agent_response'] = response.text.strip()
        
    except Exception as e:
        state['agent_response'] = f"API Error generating response: {e}"
    
    return state

# Build Graph
workflow = StateGraph(HomeState)
workflow.add_node("parse_intent", parse_intent)
workflow.add_node("fetch_memory", fetch_memory)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("parse_intent")
workflow.add_edge("parse_intent", "fetch_memory")
workflow.add_edge("fetch_memory", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()