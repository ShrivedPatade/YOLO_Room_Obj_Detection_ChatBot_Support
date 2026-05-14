from cognition.agent import app

def run():
    print("--- Smart Home Agent Initialization ---")
    
    # Setup initial blank memory
    initial_state = {
        "user_prompt": "where are the bowl, tv and the keys?",
        "target_objects": [],
        "house_memory": {},
        "agent_response": ""
    }
    
    # Run the LangGraph pipeline
    final_state = app.invoke(initial_state)
    
    print("\n--- Final Chatbot Output ---")
    print(final_state['agent_response'])
    
if __name__ == "__main__":
    run()