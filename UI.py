import os
import time
from PIL import Image
import pandas as pd
import streamlit as st

# Import your existing pipeline functions/classes from main.py / pipeline
from main import (
    Bathroom2DVisualizer,
    KohlerCPSATOptimizer,
    initialize_models,
)

# KOHLER | AI Bathroom Design & Spatial Planner
# To run:
# 1. Windows CMD: `cd "C:\Users\adit2_w4j99bx\source\repos\KOHLER" && streamlit run UI.py`
# 2. Windows PowerShell: `cd "C:\Users\adit2_w4j99bx\source\repos\KOHLER" ; streamlit run UI.py`

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM KOHLER SUSTAINABILITY THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KOHLER | AI Bathroom Design & Spatial Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kohler-inspired Sustainable Aesthetic
KOHLER_CSS = """
<style>
    /* Highlighted Text Selection: Dark Shade of Brown */
    ::-selection {
        background-color: #4A2E1B !important;
        color: #FFFFFF !important;
    }
    ::-moz-selection {
        background-color: #4A2E1B !important;
        color: #FFFFFF !important;
    }

    /* Main App Background: Set to original side panel color */
    .stApp {
        background-color: #F4EFE6 !important;
        color: #2C2A29;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Side Panel Background: One shade darker color */
    section[data-testid="stSidebar"] {
        background-color: #E8DFC8 !important;
        border-right: 1px solid #D8CEBC;
    }

    /* Side Panel Text & Headers: High-contrast Muted Dark Green */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #1E382E !important;
        font-weight: 500;
    }

    /* Target Captions, Dimensions/Price Text, & Metric Labels: Dull Dark Green (#2D4A3E) */
    .stCaption, 
    [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] .stCaption,
    .product-dim-price,
    [data-testid="stMetricLabel"] {
        color: #2D4A3E !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* Welcome Header Card */
    .welcome-card {
        background-color: #FFFFFF;
        border-left: 4px solid #2D4A3E; /* Forest Green Accent */
        padding: 24px;
        border-radius: 6px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        margin-bottom: 25px;
    }
    .welcome-title {
        color: #2D4A3E;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .welcome-subtitle {
        color: #5C554E;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Text Input Area Label: Solid Black */
    .stTextArea label p {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Text Input Area & Faint Placeholder Styling */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #C8BFA2 !important;
        border-radius: 4px !important;
        color: #2C2A29 !important;
        font-size: 14px;
    }
    .stTextArea textarea::placeholder {
        color: #8C8275 !important;
        opacity: 0.7 !important;
        font-style: italic;
    }
    .stTextArea textarea:focus {
        border-color: #2D4A3E !important;
        box-shadow: 0 0 0 1px #2D4A3E !important;
    }

    /* Form Submit Button: White Text with Dark Forest Green Background */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #2D4A3E !important;
        color: #FFFFFF !important;
        border: 1px solid #2D4A3E !important;
        border-radius: 4px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1E382E !important;
        color: #FFFFFF !important;
        border-color: #1E382E !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.12) !important;
    }

    /* Sidebar Timeline Image Cards */
    .timeline-card-latest {
        border: 2px solid #2D4A3E;
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
    .timeline-card-previous {
        border: 1px solid #C8BFA2;
        background-color: #FBF9F5;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 16px;
        opacity: 0.9;
    }
    .badge-latest {
        background-color: #2D4A3E;
        color: #FFFFFF;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .badge-history {
        background-color: #8C8275;
        color: #FFFFFF;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }

    /* Custom Dull Dark Green Line replacing standard dividers */
    hr {
        border: 0 !important;
        height: 1px !important;
        background-color: #2D4A3E !important;
        opacity: 0.5 !important;
        margin: 15px 0 !important;
    }
</style>
"""
st.markdown(KOHLER_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. STATE & BACKEND CACHING
# -----------------------------------------------------------------------------
CATALOG_PATH = "kohler_catalog.csv"

@st.cache_resource(show_spinner=False)
def load_backend():
    """
    Loads catalog data and heavy AI models into GPU/CPU memory ONCE.
    """
    if not os.path.exists(CATALOG_PATH):
        st.error(f"Catalog file not found at '{CATALOG_PATH}'. Please verify the path.")
        st.stop()
        
    catalog_df = pd.read_csv(CATALOG_PATH)
    json_extractor, encoder, rag_engine = initialize_models(catalog_df)
    return catalog_df, json_extractor, encoder, rag_engine

# Initialize session state for tracking generated layouts, history, and text form control
if "history" not in st.session_state:
    st.session_state.history = []  # List of dicts: {"timestamp": str, "prompt": str, "image_path": str, "bundle": dict}

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Load cached backend models
catalog_df, json_extractor, encoder, rag_engine = load_backend()


# -----------------------------------------------------------------------------
# 3. SIDEBAR: TIMELINE OF GENERATED LAYOUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 Generated Layouts")
    st.caption("Visual history of your architectural floorplans")
    
    if not st.session_state.history:
        st.info("No floorplans generated yet. Enter your room details to create your first design.")
    else:
        # Display images in reverse chronological order (Newest first)
        for idx, item in enumerate(reversed(st.session_state.history)):
            is_latest = (idx == 0)
            
            card_class = "timeline-card-latest" if is_latest else "timeline-card-previous"
            badge = '<span class="badge-latest">Latest Design</span>' if is_latest else f'<span class="badge-history">v{len(st.session_state.history) - idx}</span>'
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown(f"{badge} &nbsp; <small style='color:#1E382E;'>{item['timestamp']}</small>", unsafe_allow_html=True)
            
            if os.path.exists(item["image_path"]):
                st.image(item["image_path"], use_container_width=True)
            else:
                st.warning("Image file missing.")
                
            st.markdown(f"**Prompt:** *\"{item['prompt'][:60]}...\"*")
            
            # Expandable detail view inside sidebar
            with st.expander("View Products in Bundle"):
                products = item["bundle"].get("selected_products", [])
                for p in products:
                    p_name = p.get('product_name') or p.get('name') or 'Kohler Product'
                    p_price = p.get('price_usd') or p.get('price') or 0
                    p_cat = p.get('category') or p.get('product_type') or 'Product'
                    st.markdown(f"- **{p_cat.title()}**: {p_name} (${p_price:,.2f})")
            
            st.markdown('</div>', unsafe_allow_html=True)
            # Dull dark green horizontal divider line between timeline entries
            st.markdown("<hr>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. MAIN LAYOUT & USER INPUT
# -----------------------------------------------------------------------------
# Kohler Brand Header
st.markdown("<h2 style='color: #2C2A29; letter-spacing: 1px; font-weight: 300;'>KOHLER <span style='font-size: 16px; color: #2D4A3E;'>| Sustainable Design Intelligence</span></h2>", unsafe_allow_html=True)

# Welcome Banner
st.markdown("""
<div class="welcome-card">
    <div class="welcome-title">Welcome to Kohler AI Design Studio</div>
    <div class="welcome-subtitle">
        Transform your bathroom into a sustainable, modern sanctuary. Please describe your bathroom parameters below—including 
        <b>room dimensions</b> (e.g., 10x8 ft), <b>existing features or layout preferences</b>, <b>aesthetic theme</b>, and your <b>budget constraints</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# User Input Form with Faint Example Placeholder and Controlled Session State
EXAMPLE_PLACEHOLDER = "e.g., I have a 10 ft by 8 ft master bathroom. Looking for a modern minimalist style with a smart toilet, thermostatic shower system, and modern double vanity. Budget is around $5,000."

with st.form(key="design_form"):
    user_input = st.text_area(
        label="Bathroom Specifications & Requirements",
        value=st.session_state["user_input"],
        placeholder=EXAMPLE_PLACEHOLDER,
        height=140,
    )
    submit_button = st.form_submit_button(label="Generate Design & Layout ➔")


# -----------------------------------------------------------------------------
# 5. PIPELINE EXECUTION ENGINE
# -----------------------------------------------------------------------------
if submit_button:
    if not user_input.strip():
        st.warning("Please enter your bathroom details before generating.")
    else:
        with st.spinner("Processing design request through Kohler AI Engine..."):
            try:
                # Step 1: JSON Intent Extraction
                extracted_json = json_extractor.generate_json(user_input)
                
                # Retrieve parsed dimensions with explicit non-zero default fallback
                parsed_dimensions = extracted_json.get("room_dimensions")
                if not parsed_dimensions or len(parsed_dimensions) < 2 or parsed_dimensions == [0, 0]:
                    parsed_dimensions = [10.0, 8.0]

                req_products = extracted_json.get("required_products", [])

                hard_constraints = {
                    "budget": extracted_json.get("budget", 10000),
                    "required_products": req_products,
                    "room_dimensions": parsed_dimensions,
                }
                preference_sequence = extracted_json.get("preference_sequence", [])

                # Step 2 & 3: Embedding & RAG Recommendation (Expanded candidate retrieval volume to 50)
                rag_output = rag_engine.recommend(
                    hard_constraints=hard_constraints,
                    preference_sequence=preference_sequence,
                    top_k=50
                )

                # Step 4: CP-SAT Optimization
                optimizer = KohlerCPSATOptimizer(
                    rag_output=rag_output,
                    hard_constraints=hard_constraints,
                    actual_catalog=catalog_df,
                    usable_area_ratio=0.45
                )
                final_bundle = optimizer.optimize()

                # Step 5: 2D Spatial Floorplan Visualization
                selected_products = final_bundle.get("selected_products", [])
                
                if selected_products:
                    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
                    output_filename = f"layout_{timestamp_str}.png"
                    
                    visualizer = Bathroom2DVisualizer(room_dimensions=parsed_dimensions)
                    saved_path = visualizer.render_to_file(
                        products=selected_products,
                        output_filepath=output_filename
                    )

                    # Update history state
                    display_time = time.strftime("%b %d, %H:%M:%S")
                    st.session_state.history.append({
                        "timestamp": display_time,
                        "prompt": user_input,
                        "image_path": saved_path,
                        "bundle": final_bundle
                    })
                    
                    # Clear session state input and refresh view
                    st.session_state["user_input"] = ""
                    st.rerun()
                else:
                    st.error("Could not find a feasible product combination matching all strict constraints. Try increasing your budget or adjusting required products.")

            except Exception as e:
                st.error(f"An error occurred while generating the design: {e}")

# -----------------------------------------------------------------------------
# 6. MAIN PANEL RESULTS DISPLAY
# -----------------------------------------------------------------------------
if st.session_state.history:
    latest_item = st.session_state.history[-1]
    
    st.markdown("### Current Active Design Solution")
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        if os.path.exists(latest_item["image_path"]):
            st.image(
                Image.open(latest_item["image_path"]), 
                caption=f"Top-Down 2D Floorplan ({latest_item['timestamp']})",
                use_container_width=True
            )
            
    with col2:
        st.markdown("#### Selected Kohler Bundle")
        bundle = latest_item["bundle"]
        products = bundle.get("selected_products", [])
        
        # Calculate total cost from all possible price key variants
        total_cost = sum([p.get("price_usd") or p.get("price") or 0 for p in products])
        st.metric(label="Total Estimated Cost", value=f"${total_cost:,.2f}")
        
        for p in products:
            with st.container():
                p_category = p.get('category') or p.get('product_type') or 'Fixture'
                p_name = p.get('product_name') or p.get('name') or 'Kohler Product'
                
                # Retrieve price safely across schemas
                p_price = p.get('price_usd') or p.get('price') or 0.0
                
                # Retrieve dimensions safely across schemas
                if "length_ft" in p and "width_ft" in p:
                    length_val = p.get("length_ft", "N/A")
                    width_val = p.get("width_ft", "N/A")
                    dim_str = f"{length_val}' x {width_val}'"
                elif "dimensions_ft" in p and isinstance(p["dimensions_ft"], list):
                    dim_str = f"{p['dimensions_ft'][0]}' x {p['dimensions_ft'][1]}'"
                else:
                    dim_str = "Standard Scale"

                st.markdown(f"**{p_category.title()}**")
                st.markdown(f"*{p_name}*")
                # Render dimensions and price explicitly in dull dark green styling
                st.markdown(
                    f'<p class="product-dim-price">Dimensions: {dim_str} | Price: ${p_price:,.2f}</p>', 
                    unsafe_allow_html=True
                )
                st.markdown("<hr>", unsafe_allow_html=True)