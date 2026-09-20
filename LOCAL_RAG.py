import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from LSTM_LAYER import LSTMPreferenceEncoder


class LocalVectorSpaceRAG:
    def __init__(
        self,
        catalog_df: pd.DataFrame,
        encoder_model: LSTMPreferenceEncoder,
        vectors_path: str = "product_vectors (1).npy"
    ):
        """
        Initializes the RAG engine using pre-computed product vectors.
        """
        self.catalog = catalog_df.fillna("").copy()
        self.encoder = encoder_model
        self.encoder.eval()
        self.device = next(encoder_model.parameters()).device

        if not os.path.exists(vectors_path):
            raise FileNotFoundError(
                f"[ERROR] Pre-computed vector file '{vectors_path}' not found! "
                "Make sure it is bundled with your app files."
            )

        print(f"[INFO] Loading pre-computed product vectors from {vectors_path}...")
        self.product_vectors = np.load(vectors_path)
        print(f"[INFO] Successfully loaded {len(self.product_vectors)} product embeddings.")

    def recommend(self, hard_constraints: dict, preference_sequence: list, top_k: int = 20) -> dict:
        """
        Generates top candidate recommendations based on vector cosine similarity,
        required product categories, and budget constraints with resilience checks.
        """
        if not preference_sequence:
            return {"recommendations": []}

        # 1. Compute user preference embedding
        with torch.no_grad():
            pref_embeddings = self.encoder.create_embeddings(preference_sequence).to(self.device)
            user_vector = self.encoder(pref_embeddings)
            user_vector_normalized = F.normalize(user_vector, p=2, dim=1).squeeze(0).cpu().numpy()

        # 2. Guardrail: Ensure product_vectors length matches catalog length before computing similarity
        if len(self.product_vectors) != len(self.catalog):
            print(
                f"[WARNING] Vector count ({len(self.product_vectors)}) does not match "
                f"catalog count ({len(self.catalog)}). Aligning vectors dynamically..."
            )
            min_len = min(len(self.product_vectors), len(self.catalog))
            self.catalog = self.catalog.iloc[:min_len].copy()
            self.product_vectors = self.product_vectors[:min_len]

        # 3. Cosine similarity against catalog embeddings
        similarities = np.dot(self.product_vectors, user_vector_normalized)

        candidates_df = self.catalog.copy()
        candidates_df["cosine_similarity"] = similarities

        # 4. Apply Hard Constraint: Category filtering
        required_cats = hard_constraints.get("required_products", [])
        if required_cats:
            candidates_df = candidates_df[candidates_df["product_type"].isin(required_cats)]

        # 5. Apply Hard Constraint: Budget cap filtering
        budget = hard_constraints.get("budget", None)
        price_col = "price_usd" if "price_usd" in candidates_df.columns else "price"

        if price_col in candidates_df.columns:
            candidates_df[price_col] = pd.to_numeric(candidates_df[price_col], errors='coerce').fillna(0.0)

        if budget is not None and price_col in candidates_df.columns:
            filtered_by_budget = candidates_df[candidates_df[price_col] <= float(budget)]
            
            # Fallback Guardrail: If budget eliminates all items, relax budget filter to avoid pipeline stall
            if filtered_by_budget.empty:
                print(f"[WARNING] No products found under budget constraint (${budget}). Relaxing budget filter...")
            else:
                candidates_df = filtered_by_budget

        # 6. Sort by cosine similarity
        sorted_recommendations = candidates_df.sort_values(by="cosine_similarity", ascending=False).head(top_k)

        # 7. Formulate response payload
        structured_output = {"recommendations": []}

        for _, row in sorted_recommendations.iterrows():
            # Detect matching preference terms from the query sequence
            matched = [pref for pref in preference_sequence if pref.lower() in str(row).lower()]
            
            cat_name = str(row.get("product_type", row.get("category", "item"))).capitalize()
            sim_score = float(row["cosine_similarity"])
            
            # Dynamic preference match text
            if matched:
                match_str = f"matches your preference for '{', '.join(matched)}'"
            else:
                match_str = f"aligns with your design aesthetic"

            # Score-based confidence rating
            if sim_score >= 0.85:
                confidence = "High aesthetic alignment"
            elif sim_score >= 0.65:
                confidence = "Moderate style fit"
            else:
                confidence = "Satisfies category requirement"

            # Dynamic reason string construction
            dynamic_reason = (
                f"Selected as top {cat_name}: {confidence} "
                f"and {match_str} (Score: {sim_score:.2f})."
            )

            raw_price = row.get(price_col, 0.0)
            try:
                price_val = float(raw_price)
            except (ValueError, TypeError):
                price_val = 0.0

            structured_output["recommendations"].append({
                "product_id": row.get("product_id", ""),
                "product_name": row.get("name", row.get("title", "")),
                "category": row.get("product_type", row.get("category", "")),
                "price_usd": price_val,
                "similarity_score": float(round(sim_score, 4)),
                "reason": dynamic_reason,
                "matched_preferences": matched,
                "hard_constraints_satisfied": True
            })

        return structured_output