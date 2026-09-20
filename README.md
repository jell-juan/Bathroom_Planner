# KOHLER AI Bathroom Designer & Planner

An intelligent, spatial-aware AI design assistant built for **Track 1 of the KOHLER-MITWPU AI Research Lab Case Study**. The system combines Natural Language Processing, Sequential Preference Encoders, Vector RAG Search, and Deterministic CP-SAT Constraint Optimization to recommend fully compatible, budget-optimized, and spatially feasible Kohler bathroom product bundles.

---

## Key Features

- **Natural Language Intent Extraction:** Parses user constraints (room dimensions, budget limits, required product categories, and design styles) into structured JSON.
- **Sequential Preference Encoding:** Utilizes a trained LSTM model to capture ordering and aesthetic nuance from natural language design descriptions.
- **Vector Space RAG:** Reranks product recommendations based on aesthetic cosine similarity in embedding space.
- **CP-SAT Spatial & Budget Optimization:** Enforces strict physical room bounds and budget limits via Constraint Programming to guarantee no component overlaps or budget overruns.
- **Interactive 2D Visualization & Timeline:** Displays 2D floorplan layouts with multi-turn design history tracking in a custom Streamlit UI.

---

## Architecture Overview

```text
[ User Input Query ]
│
▼
┌───────────────────────────────┐
│        NLP Layer (LLM)        │ ──► Extract JSON Requirements
└───────────────┬───────────────┘
│
┌───────┴────────┐
▼                ▼
┌──────────────┐  ┌──────────────┐
│  LSTM Layer  │  │  Vector RAG  │
└───────┬──────┘  └───────┬──────┘
│                │
└───────┬────────┘
▼
┌───────────────────────────────┐
│     CP-SAT Solver Engine      │ ──► Hard Constraints (Budget & Dimensions)
└───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│ Streamlit UI & 2D Visualizer  │ ──► Multi-turn floorplan & budget output
└───────────────────────────────┘
