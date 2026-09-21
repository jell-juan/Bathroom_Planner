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

    def _safe_float(self, val, default=0.0):
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def recommend(self, hard_constraints: dict, preference_sequence: list, top_k: int = 20) -> dict:
        """
        Generates candidate recommendations using vector similarity, category filters, 
        and budget constraint fallback rules.
        """
        # Align catalog and vector lengths if mismatched
        if len(self.product_vectors) != len(self.catalog):
            print(
                f"[WARNING] Vector count ({len(self.product_vectors)}) does not match "
                f"catalog count ({len(self.catalog)}). Aligning vectors dynamically..."
            )
            min_len = min(len(self.product_vectors), len(self.catalog))
            self.catalog = self.catalog.iloc[:min_len].copy()
            self.product_vectors = self.product_vectors[:min_len]

        # 1. Calculate Vector Embedding (or fallback to neutral vector if sequence is empty)
        if preference_sequence:
            with torch.no_grad():
                pref_embeddings = self.encoder.create_embeddings(preference_sequence).to(self.device)
                user_vector = self.encoder(pref_embeddings)
                user_vector_normalized = F.normalize(user_vector, p=2, dim=1).squeeze(0).cpu().numpy()
        else:
            # Fallback: Use mean vector across all catalog items if no style keywords exist
            user_vector_normalized = np.mean(self.product_vectors, axis=0)
            norm = np.linalg.norm(user_vector_normalized)
            if norm > 0:
                user_vector_normalized /= norm

        # 2. Compute Cosine Similarity
        similarities = np.dot(self.product_vectors, user_vector_normalized)
        candidates_df = self.catalog.copy()
        candidates_df["cosine_similarity"] = similarities

        # 3. Flexible Category Filtering
        required_cats = hard_constraints.get("required_products", [])
        if required_cats:
            # Determine correct column name dynamically
            cat_col = "product_type" if "product_type" in candidates_df.columns else "category"
            
            if cat_col in candidates_df.columns:
                req_cats_clean = [str(c).lower().rstrip('s') for c in required_cats]
                
                # Broad/flexible matching (case-insensitive & plural invariant)
                cat_mask = candidates_df[cat_col].astype(str).apply(
                    lambda val: any(req_cat in val.lower().rstrip('s') for req_cat in req_cats_clean)
                )
                
                filtered_cats = candidates_df[cat_mask]
                if not filtered_cats.empty:
                    candidates_df = filtered_cats
                else:
                    print(f"[WARNING] No products matched category requirement {required_cats}. Retaining all categories.")

        # 4. Budget Constraint Filtering with Fallback
        budget = hard_constraints.get("budget", None)
        price_col = "price_usd" if "price_usd" in candidates_df.columns else "price"

        if price_col in candidates_df.columns:
            candidates_df[price_col] = pd.to_numeric(candidates_df[price_col], errors='coerce').fillna(0.0)

        safe_budget = self._safe_float(budget, default=None) if budget is not None else None

        if safe_budget is not None and safe_budget > 0 and price_col in candidates_df.columns:
            filtered_by_budget = candidates_df[candidates_df[price_col] <= safe_budget]
            
            if filtered_by_budget.empty:
                print(f"[WARNING] No products found under budget (${safe_budget}). Relaxing budget filter...")
            else:
                candidates_df = filtered_by_budget

        # 5. Sort Candidates
        sorted_recommendations = candidates_df.sort_values(by="cosine_similarity", ascending=False).head(top_k)

        # 6. Formulate Output Payload
        structured_output = {"recommendations": []}

        for _, row in sorted_recommendations.iterrows():
            matched = [pref for pref in preference_sequence if pref.lower() in str(row).lower()]
            
            cat_col = "product_type" if "product_type" in row else "category"
            cat_name = str(row.get(cat_col, "item")).capitalize()
            sim_score = float(row.get("cosine_similarity", 0.5))
            
            if matched:
                match_str = f"matches your preference for '{', '.join(matched)}'"
            else:
                match_str = "aligns with your overall setup"

            dynamic_reason = f"Selected as candidate {cat_name}: {match_str} (Score: {sim_score:.2f})."

            price_val = self._safe_float(row.get(price_col, 0.0), default=0.0)
            len_ft = self._safe_float(row.get("length_ft", row.get("length", None)), default=None)
            wid_ft = self._safe_float(row.get("width_ft", row.get("width", None)), default=None)

            structured_output["recommendations"].append({
                "product_id": str(row.get("product_id", "")),
                "product_name": str(row.get("name", row.get("title", row.get("product_name", "")))),
                "category": str(row.get(cat_col, "")),
                "price_usd": price_val,
                "length_ft": len_ft,
                "width_ft": wid_ft,
                "dimensions_ft": [len_ft, wid_ft] if (len_ft and wid_ft) else None,
                "similarity_score": float(round(sim_score, 4)),
                "reason": dynamic_reason,
                "matched_preferences": matched,
                "hard_constraints_satisfied": True
            })

        return structured_output