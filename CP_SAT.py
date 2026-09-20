import json
import pandas as pd
from ortools.sat.python import cp_model


class KohlerCPSATOptimizer:

    def __init__(
        self,
        rag_output: dict,
        hard_constraints: dict,
        actual_catalog: pd.DataFrame,
        usable_area_ratio: float = 0.40,  # Max 40% room floor occupancy
    ):
        self.rag_output = rag_output
        self.budget = hard_constraints.get("budget", None)
        self.required_categories = hard_constraints.get("required_products", [])
        self.room_dimensions = hard_constraints.get("room_dimensions", None)  # [length_ft, width_ft]
        self.candidates = rag_output.get("recommendations", [])
        self.actual_catalog = actual_catalog.copy()
        self.usable_area_ratio = usable_area_ratio

        self.df = pd.DataFrame(self.candidates)

    def _extract_dimensions_sqft(
        self, catalog_item: pd.Series
    ) -> tuple[float, float, float]:
        """Extracts product dimensions with fallback across all standard catalog naming conventions."""
        # 1. Flexible key lookup across common CSV column names
        len_keys = ["length_inches", "length_in", "length", "depth_inches", "depth_in", "depth"]
        wid_keys = ["width_inches", "width_in", "width", "span_inches", "span"]

        p_len_in = None
        p_wid_in = None

        for k in len_keys:
            if k in catalog_item and pd.notna(catalog_item[k]):
                p_len_in = catalog_item[k]
                break

        for k in wid_keys:
            if k in catalog_item and pd.notna(catalog_item[k]):
                p_wid_in = catalog_item[k]
                break

        # Defaults if dimensions are missing in CSV (24in x 18in fallback)
        if p_len_in is None or p_len_in == 0:
            p_len_in = 24.0
        if p_wid_in is None or p_wid_in == 0:
            p_wid_in = 18.0

        try:
            p_len_ft = float(p_len_in) / 12.0
            p_wid_ft = float(p_wid_in) / 12.0
        except (ValueError, TypeError):
            p_len_ft, p_wid_ft = 2.0, 1.5  # Fallback: 2ft x 1.5ft

        area_sqft = p_len_ft * p_wid_ft
        return p_len_ft, p_wid_ft, area_sqft

    def optimize(self) -> dict:
        if self.df.empty:
            print("[WARNING] No candidates passed to CP-SAT Optimizer.")
            return {"status": "No Candidates", "selected_products": []}

        model = cp_model.CpModel()

        # Step 1: Decision Variables
        x = {}
        for i, row in self.df.iterrows():
            x[i] = model.NewBoolVar(f"select_{row['product_id']}_{row['category']}")

        # Step 2: Category Matching (At most 1 per category, exactly 1 for required categories present)
        for category in set(self.df["category"].unique()):
            cat_indices = self.df[self.df["category"] == category].index.tolist()
            if category in self.required_categories:
                # Require exactly 1 product from this category
                model.Add(sum(x[idx] for idx in cat_indices) == 1)
            else:
                # Optional categories: select at most 1
                model.Add(sum(x[idx] for idx in cat_indices) <= 1)

        # Step 3: Extract Prices and Dimensions
        prices = []
        product_dims = []

        for _, row in self.df.iterrows():
            matched_item = self.actual_catalog[
                self.actual_catalog["product_id"] == row["product_id"]
            ]

            if not matched_item.empty:
                item_series = matched_item.iloc[0]

                # Price lookup with fallbacks
                price_val = item_series.get("price_usd", item_series.get("price", 0.0))
                try:
                    prices.append(int(float(price_val)))
                except (ValueError, TypeError):
                    prices.append(100)

                dims = self._extract_dimensions_sqft(item_series)
                product_dims.append(dims)
            else:
                prices.append(100)
                product_dims.append((2.0, 1.5, 3.0))

        # Step 4: Budget Limit
        if self.budget is not None and self.budget > 0:
            model.Add(
                sum(x[i] * prices[i] for i in range(len(self.df))) <= int(self.budget)
            )

        # Step 5: Spatial Feasibility
        room_area_sqft = None
        if self.room_dimensions and len(self.room_dimensions) == 2:
            room_len_ft = float(self.room_dimensions[0])
            room_wid_ft = float(self.room_dimensions[1])
            room_area_sqft = room_len_ft * room_wid_ft

            max_usable_area_sqft = room_area_sqft * self.usable_area_ratio

            for i in range(len(self.df)):
                p_len_ft, p_wid_ft, _ = product_dims[i]

                fits_normal = (p_len_ft <= room_len_ft) and (p_wid_ft <= room_wid_ft)
                fits_rotated = (p_wid_ft <= room_len_ft) and (p_len_ft <= room_wid_ft)

                # If item physically exceeds room boundaries in both orientations, disable it
                if not (fits_normal or fits_rotated):
                    model.Add(x[i] == 0)

            # Footprint Area Limit
            scaled_areas = [int(dims[2] * 100) for dims in product_dims]
            scaled_max_area = int(max_usable_area_sqft * 100)

            model.Add(
                sum(x[i] * scaled_areas[i] for i in range(len(self.df)))
                <= scaled_max_area
            )

        # Step 6: Objective Function (Maximize Similarity)
        scale_factor = 10000
        similarities = [
            int(row.get("similarity_score", 0.5) * scale_factor)
            for _, row in self.df.iterrows()
        ]
        model.Maximize(sum(x[i] * similarities[i] for i in range(len(self.df))))

        # Step 7: Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        # Step 8: Parse Results
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected_items = []
            total_cost = 0
            total_area_used = 0.0

            for i, row in self.df.iterrows():
                if solver.Value(x[i]) == 1:
                    p_len, p_wid, p_area = product_dims[i]
                    selected_items.append({
                        "product_id": row["product_id"],
                        "product_name": row["product_name"],
                        "category": row["category"],
                        "price_usd": prices[i],
                        "dimensions_ft": [round(p_len, 2), round(p_wid, 2)],
                        "length_ft": round(p_len, 2),
                        "width_ft": round(p_wid, 2),
                        "area_sqft": round(p_area, 2),
                        "similarity_score": row.get("similarity_score", 0.0),
                        "reason": row.get("reason", "Optimal candidate selection"),
                    })
                    total_cost += prices[i]
                    total_area_used += p_area

            return {
                "status": "Optimal Bundle Found",
                "total_cost": total_cost,
                "budget_limit": self.budget,
                "spatial_metrics": {
                    "room_dimensions_ft": self.room_dimensions,
                    "total_room_area_sqft": round(room_area_sqft, 2) if room_area_sqft else None,
                    "used_area_sqft": round(total_area_used, 2),
                    "usable_area_limit_sqft": round(room_area_sqft * self.usable_area_ratio, 2) if room_area_sqft else None,
                },
                "selected_products": selected_items,
            }
        else:
            return {
                "status": "Infeasible - Constraints could not be satisfied simultaneously.",
                "selected_products": [],
            }