from NLP_LAYER import SmolLMJSONExtractor

def merge_requirements(current_state: dict, turn_extraction: dict) -> dict:
    """
    Accumulates requirements across turns without overwriting previous selections.
    """
    # Preserve or update budget
    if turn_extraction.get("budget") is not None:
        current_state["budget"] = turn_extraction["budget"]

    # Preserve or update dimensions
    if turn_extraction.get("room_dimensions") is not None:
        current_state["room_dimensions"] = turn_extraction["room_dimensions"]

    # Append new required products without duplicates
    for prod in turn_extraction.get("required_products", []):
        if prod not in current_state["required_products"]:
            current_state["required_products"].append(prod)

    # Append new design preferences without duplicates
    for pref in turn_extraction.get("preference_sequence", []):
        if pref not in current_state["preference_sequence"]:
            current_state["preference_sequence"].append(pref)

    return current_state


def run_pipeline():
    extractor = SmolLMJSONExtractor()

    # Active session state carried across dialogue turns
    session_state = {
        "budget": None,
        "room_dimensions": None,
        "required_products": [],
        "preference_sequence": []
    }

    print("\n--- Bathroom Design Assistant Ready ---")
    
    while True:
        user_input = input("\nDescribe your bathroom requirements (or type 'exit' to quit):\n> ")
        
        if user_input.strip().lower() == "exit":
            print("Exiting pipeline. Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            # 1. Extract intent from current turn
            turn_json = extractor.generate_json(user_input)

            # 2. Accumulate turn intent into global conversation state
            session_state = merge_requirements(session_state, turn_json)

            print("\n--- Extracted Cumulative JSON ---")
            print(session_state)

            # Pass session_state downstream to your RAG/Optimization engine here...

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed on current input: {e}")
            print("Please try rephrasing your request.")

if __name__ == "__main__":
    run_pipeline()