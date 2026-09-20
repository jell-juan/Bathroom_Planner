import os
import pprint
import torch
import pandas as pd

from CP_SAT import KohlerCPSATOptimizer
from LOCAL_RAG import LocalVectorSpaceRAG
from LSTM_LAYER import LSTMPreferenceEncoder
from NLP_LAYER import SmolLMJSONExtractor
from visualizer_2d import Bathroom2DVisualizer

# ==========================================================
# GLOBAL CONFIGURATION & PATH RESOLUTION
# ==========================================================

torch.set_num_threads(4)  # Match CPU physical core count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_CHECKPOINT = os.path.join(SCRIPT_DIR, "lstm_preference_encoder (1).pth")
CATALOG_PATH = os.path.join(SCRIPT_DIR, "kohler_catalog.csv")
VECTORS_PATH = os.path.join(SCRIPT_DIR, "product_vectors (1).npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# STATEFUL MANAGEMENT HELPERS
# ==========================================================

def merge_requirements(current_state: dict, turn_extraction: dict) -> dict:
    """
    Accumulates requirements across interaction turns without wiping previous context.
    """
    if turn_extraction.get("budget") is not None:
        current_state["budget"] = turn_extraction["budget"]

    if turn_extraction.get("room_dimensions") is not None:
        current_state["room_dimensions"] = turn_extraction["room_dimensions"]

    for prod in turn_extraction.get("required_products", []):
        if prod not in current_state["required_products"]:
            current_state["required_products"].append(prod)

    for pref in turn_extraction.get("preference_sequence", []):
        if pref not in current_state["preference_sequence"]:
            current_state["preference_sequence"].append(pref)

    return current_state


# ==========================================================
# MODEL INITIALIZATION
# ==========================================================

def initialize_models(catalog_df: pd.DataFrame):
    """
    Loads all required models into memory once.
    """
    print("[INFO] Initializing SmolLM JSON Extractor...")
    json_extractor = SmolLMJSONExtractor()

    print("[INFO] Initializing LSTM Preference Encoder...")
    encoder = LSTMPreferenceEncoder()
    state_dict = torch.load(LSTM_CHECKPOINT, map_location=DEVICE)
    encoder.load_state_dict(state_dict)
    encoder.to(DEVICE)
    encoder.eval()
    print("[INFO] LSTM checkpoint loaded successfully.")

    print("[INFO] Initializing Local Vector Space RAG Engine...")
    rag_engine = LocalVectorSpaceRAG(
        catalog_df=catalog_df,
        encoder_model=encoder,
        vectors_path=VECTORS_PATH
    )

    return json_extractor, encoder, rag_engine


# ==========================================================
# END-TO-END PIPELINE LOOP
# ==========================================================

def run_interactive_pipeline(catalog_df: pd.DataFrame):
    """
    Stateful interactive pipeline with RAG, CP-SAT optimization, and 2D visualizer.
    """
    print("\n[INFO] Booting up KOHLER Design System...")
    
    # Initialize models ONCE before entering dialogue loop
    json_extractor, encoder, rag_engine = initialize_models(catalog_df)
    
    # Global state dictionary retained across interactive turns
    session_state = {
        "budget": None,
        "room_dimensions": None,
        "required_products": [],
        "preference_sequence": []
    }

    print("\n[INFO] System Ready. Enter your requirements below.")

    while True:
        user_input = input("\nDescribe your bathroom requirements (or type 'exit' to quit):\n> ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exiting pipeline. Goodbye!")
            break
            
        if not user_input.strip():
            continue

        try:
            # ---------------------------------------
            # Step 1: JSON Extraction & State Merge
            # ---------------------------------------
            turn_extraction = json_extractor.generate_json(user_input)
            session_state = merge_requirements(session_state, turn_extraction)

            print("\n--- Cumulative Extracted Requirements ---")
            pprint.pprint(session_state)

            # Fallback room dimensions if unmentioned (Default: 10 ft x 8 ft)
            parsed_dimensions = session_state.get("room_dimensions") or [10.0, 8.0]

            hard_constraints = {
                "budget": session_state.get("budget"),
                "required_products": session_state.get("required_products", []),
                "room_dimensions": parsed_dimensions,
            }

            preference_sequence = session_state.get("preference_sequence", [])

            # ---------------------------------------
            # Step 2: RAG Candidate Retrieval
            # ---------------------------------------
            rag_output = rag_engine.recommend(
                hard_constraints=hard_constraints,
                preference_sequence=preference_sequence,
                top_k=20
            )

            print("\n--- RAG Candidates ---")
            print(f"Retrieved {len(rag_output.get('recommendations', []))} candidates")

            # ---------------------------------------
            # Step 3: CP-SAT Bundle Optimization
            # ---------------------------------------
            optimizer = KohlerCPSATOptimizer(
                rag_output=rag_output,
                hard_constraints=hard_constraints,
                actual_catalog=catalog_df
            )

            final_bundle = optimizer.optimize()

            print("\n--- Final Optimized Bundle ---")
            pprint.pprint(final_bundle)

            # ---------------------------------------
            # Step 4: 2D Spatial Floorplan Visualization
            # ---------------------------------------
            selected_products = final_bundle.get("selected_products", [])
            
            if selected_products:
                print("\n[INFO] Rendering 2D Floorplan Layout...")
                visualizer = Bathroom2DVisualizer(room_dimensions=parsed_dimensions)
                
                output_image_path = visualizer.render_to_file(
                    products=selected_products, 
                    output_filepath="bathroom_2d_layout.png"
                )
                print(f"[SUCCESS] 2D layout generated and saved to: {output_image_path}")
            else:
                print("[WARNING] Skipping visualizer: No products found in optimized bundle.")
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed on current input: {e}")
            print("Please try rephrasing your request.")


def process_single_request(
    user_input: str,
    catalog_df: pd.DataFrame,
    json_extractor,
    rag_engine,
    output_image_path: str = "bathroom_2d_layout.png"
) -> dict:
    """
    Executes a single end-to-end pipeline pass without entering a CLI terminal loop.
    Designed for UI integration (e.g., Streamlit, FastAPI, Gradio).
    """
    # ---------------------------------------
    # Step 1: JSON Intent Extraction
    # ---------------------------------------
    extracted_json = json_extractor.generate_json(user_input)
    parsed_dimensions = extracted_json.get("room_dimensions") or [10.0, 8.0]

    hard_constraints = {
        "budget": extracted_json.get("budget"),
        "required_products": extracted_json.get("required_products", []),
        "room_dimensions": parsed_dimensions,
    }
    preference_sequence = extracted_json.get("preference_sequence", [])

    # ---------------------------------------
    # Step 2: RAG Candidate Retrieval
    # ---------------------------------------
    rag_output = rag_engine.recommend(
        hard_constraints=hard_constraints,
        preference_sequence=preference_sequence,
        top_k=20
    )

    # ---------------------------------------
    # Step 3: CP-SAT Optimization
    # ---------------------------------------
    optimizer = KohlerCPSATOptimizer(
        rag_output=rag_output,
        hard_constraints=hard_constraints,
        actual_catalog=catalog_df
    )
    final_bundle = optimizer.optimize()

    # ---------------------------------------
    # Step 4: 2D Spatial Floorplan Visualization
    # ---------------------------------------
    selected_products = final_bundle.get("selected_products", [])
    saved_image_path = None

    if selected_products:
        visualizer = Bathroom2DVisualizer(room_dimensions=parsed_dimensions)
        saved_image_path = visualizer.render_to_file(
            products=selected_products,
            output_filepath=output_image_path
        )

    return {
        "extracted_json": extracted_json,
        "final_bundle": final_bundle,
        "saved_image_path": saved_image_path,
        "room_dimensions": parsed_dimensions
    }

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    if not os.path.exists(CATALOG_PATH):
        print(f"[ERROR] Could not find catalog at: {CATALOG_PATH}")
    else:
        catalog_df = pd.read_csv(CATALOG_PATH)
        run_interactive_pipeline(catalog_df)