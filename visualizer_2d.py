import io
import base64
import os
import urllib.request
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


@lru_cache(maxsize=32)
def load_image_cached(image_path: str) -> Optional[Image.Image]:
    """Caches local or remote product images in RAM to prevent blocking HTTP requests."""
    if not image_path:
        return None
        
    try:
        if os.path.exists(str(image_path)):
            return Image.open(image_path)
            
        if str(image_path).startswith(('http://', 'https://')):
            req = urllib.request.Request(image_path, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=0.3) as response:
                image_data = response.read()
                return Image.open(io.BytesIO(image_data))
    except Exception:
        pass  # Graceful fallback on HTTP/file timeout
        
    return None


class Bathroom2DVisualizer:
    """Production-ready 2D floorplan visualizer[cite: 5]."""
    
    CATEGORY_CONFIG = {
        "shower": {"color": "#E3F2FD", "edge": "#1E88E5", "anchor": "corner_top_left"},
        "tub": {"color": "#E0F7FA", "edge": "#00ACC1", "anchor": "corner_bottom_left"},
        "vanity": {"color": "#E8F5E9", "edge": "#43A047", "anchor": "wall_top"},
        "sink": {"color": "#E8F5E9", "edge": "#43A047", "anchor": "wall_top"},
        "faucet": {"color": "#FFF3E0", "edge": "#FB8C00", "anchor": "wall_top"},
        "toilet": {"color": "#F3E5F5", "edge": "#8E24AA", "anchor": "wall_bottom"},
        "mirror": {"color": "#FFFDE7", "edge": "#FDD835", "anchor": "wall_top"}
    }

    def __init__(self, room_dimensions: Tuple[float, float], dpi: int = 80, fast_vector_mode: bool = False):
        # Sanity check dimensions to prevent zero/massive scale bugs
        length = float(room_dimensions[0]) if room_dimensions and room_dimensions[0] else 10.0
        width = float(room_dimensions[1]) if room_dimensions and room_dimensions[1] else 10.0
        
        # Clamp dimensions between 4ft and 50ft
        self.room_length = min(max(length, 4.0), 50.0)
        self.room_width = min(max(width, 4.0), 50.0)
        self.dpi = dpi
        self.fast_vector_mode = fast_vector_mode

    def _extract_dimensions(self, item: Dict[str, Any]) -> Tuple[float, float]:
        p_len = item.get("length_ft")
        p_wid = item.get("width_ft")

        if (p_len is None or p_len == 0) and "dimensions_ft" in item:
            dims = item.get("dimensions_ft")
            if isinstance(dims, (list, tuple)) and len(dims) >= 2:
                p_len, p_wid = dims[0], dims[1]

        if not p_len or p_len == 0 or not p_wid or p_wid == 0:
            category = str(item.get("category", "")).lower()
            defaults = {
                "sink": (2.0, 1.5),
                "vanity": (4.0, 2.0),
                "toilet": (2.5, 1.5),
                "shower": (3.5, 3.5),
                "tub": (5.0, 2.5),
                "mirror": (2.5, 2.0),
                "faucet": (1.0, 1.0)
            }
            p_len, p_wid = defaults.get(category, (2.0, 2.0))

        p_len = max(0.5, min(float(p_len), self.room_length - 0.5))
        p_wid = max(0.5, min(float(p_wid), self.room_width - 0.5))
        
        return p_len, p_wid

    def _determine_item_positions(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        placed_products = []
        margin = 0.5
        spacing = 0.5
        top_wall_x = margin
        bottom_wall_x = margin
        left_wall_y = margin
        max_x = self.room_length - margin

        for item in products:
            p_type = str(item.get("category", item.get("product_type", "fixture"))).lower()
            p_len, p_wid = self._extract_dimensions(item)
            config = self.CATEGORY_CONFIG.get(p_type, {"color": "#ECEFF1", "edge": "#546E7A", "anchor": "center"})
            anchor = config["anchor"]

            if anchor == "corner_top_left":
                x = margin
                y = self.room_width - p_wid - margin
                top_wall_x = max(top_wall_x, x + p_len + spacing)
            elif anchor == "corner_bottom_left":
                x = margin
                y = margin
                bottom_wall_x = max(bottom_wall_x, x + p_len + spacing)
            elif anchor == "wall_top":
                if top_wall_x + p_len > max_x:
                    top_wall_x = margin
                x = top_wall_x
                y = self.room_width - p_wid - margin
                top_wall_x += p_len + spacing
            elif anchor == "wall_bottom":
                if bottom_wall_x + p_len > max_x:
                    bottom_wall_x = margin
                x = bottom_wall_x
                y = margin
                bottom_wall_x += p_len + spacing
            else:
                x = margin
                y = left_wall_y
                left_wall_y += p_wid + spacing

            x = max(0.2, min(x, self.room_length - p_len - 0.2))
            y = max(0.2, min(y, self.room_width - p_wid - 0.2))

            placed_products.append({
                "info": item,
                "x": x, "y": y,
                "len": p_len, "wid": p_wid,
                "type": p_type,
                "config": config
            })

        return placed_products

    def generate_figure(self, products: List[Dict[str, Any]]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
        
        # Room boundary patch
        room_bg = patches.Rectangle(
            (0, 0), self.room_length, self.room_width,
            linewidth=2.5, edgecolor='#212121', facecolor='#FAFAFA', zorder=1
        )
        ax.add_patch(room_bg)
        
        # Grid lines
        step_x = max(1, int(self.room_length // 10))
        step_y = max(1, int(self.room_width // 10))
        ax.set_xticks(range(0, int(self.room_length) + 1, step_x))
        ax.set_yticks(range(0, int(self.room_width) + 1, step_y))
        ax.grid(True, which='both', color='#E0E0E0', linestyle='--', linewidth=0.5, zorder=2)

        placed_items = self._determine_item_positions(products)

        for item in placed_items:
            x, y = item["x"], item["y"]
            w, h = item["len"], item["wid"]
            info = item["info"]
            config = item["config"]
            img = None

            if not self.fast_vector_mode:
                img_path = info.get("image_path", info.get("image", None))
                if img_path:
                    img = load_image_cached(img_path)
            
            if img is not None:
                try:
                    ax.imshow(img, extent=[x, x + w, y, y + h], zorder=4, aspect='auto')
                    rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor=config["edge"], facecolor='none', zorder=5)
                    ax.add_patch(rect)
                except Exception:
                    img = None

            if img is None:
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=1.2,
                    edgecolor=config["edge"], facecolor=config["color"],
                    alpha=0.85, zorder=3
                )
                ax.add_patch(rect)

            name = info.get("product_name", info.get("name", item["type"].capitalize()))
            display_label = f"{name[:18]}\n({w:.1f}' x {h:.1f}')"
            
            ax.text(
                x + w / 2.0, y + h / 2.0, display_label,
                color='#111111', weight='bold', fontsize=6.0,
                ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='#CCCCCC', lw=0.5)
            )

        ax.set_xlim(-0.5, self.room_length + 0.5)
        ax.set_ylim(-0.5, self.room_width + 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"2D Layout Plan ({self.room_length:.1f}' x {self.room_width:.1f}')", fontsize=10, fontweight='bold')
        ax.axis('off')

        plt.tight_layout(pad=0.5)
        return fig

    def render_to_file(self, products: List[Dict[str, Any]], output_filepath: str = "bathroom_2d_layout.png") -> str:
        fig = self.generate_figure(products)
        fig.savefig(output_filepath, bbox_inches='tight', dpi=self.dpi)
        plt.close(fig)  # Prevents memory accumulation across runs[cite: 4]
        return output_filepath