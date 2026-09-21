import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sentence_transformers import SentenceTransformer

# =====================================================================
# LSTM Preference Encoder
# =====================================================================
class LSTMPreferenceEncoder(nn.Module):
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False
    ):
        super(LSTMPreferenceEncoder, self).__init__()

        # Initialize the Sentence Transformer
        print(f"[INFO] Loading embedding model '{embedding_model_name}'...")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # LSTM configuration
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,  # Input expected as (batch, seq_len, features)
            bidirectional=self.bidirectional
        )
        self.projection = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            128
        )
        self.product_projection = nn.Linear(
            embedding_dim,
            128
        )

    def create_embeddings(self, preferences: list) -> torch.Tensor:
        if not preferences:
            return torch.zeros((1, 1, self.embedding_dim))

        embeddings_np = self.embedding_model.encode(preferences)
        embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)
        return embeddings_tensor.unsqueeze(0)

    def forward(self, preference_embeddings: torch.Tensor) -> torch.Tensor:
        lstm_out, (hn, cn) = self.lstm(preference_embeddings)
        final_vector = lstm_out[:, -1, :]
        final_vector = self.projection(final_vector)
        return final_vector

    def fit_from_catalog(self, catalog_csv_path: str, save_checkpoint_path: str, epochs: int = 10, lr: float = 1e-3):
        """
        Trains the projection/LSTM weights using self-supervised contrastive learning
        from textual features in kohler_catalog.csv when no checkpoint exists.
        """
        print(f"[INFO] Checkpoint not found. Auto-training LSTM Encoder from '{catalog_csv_path}'...")
        
        if not os.path.exists(catalog_csv_path):
            raise FileNotFoundError(f"[ERROR] Catalog CSV file '{catalog_csv_path}' not found!")

        df = pd.read_csv(catalog_csv_path).fillna("")
        
        # Extract descriptive text columns to construct preference sentences
        text_data = []
        for _, row in df.iterrows():
            combined_text = " ".join([
                str(row.get(col, "")) for col in ["product_type", "style", "finish", "description", "title"]
                if col in row and str(row[col])
            ])
            if combined_text.strip():
                text_data.append(combined_text.strip())

        if not text_data:
            text_data = ["standard modern fixture", "matte black bathroom finish", "contemporary style unit"]

        # Encode text data into sequence representations
        print(f"[INFO] Encoding {len(text_data)} catalog descriptions for training...")
        embeddings_np = self.embedding_model.encode(text_data)
        embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)

        # Self-supervised contrastive optimization loop
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        device = next(self.parameters()).device

        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(embeddings_tensor), 32):  # batch_size=32
                batch = embeddings_tensor[i:i+32].to(device)
                if batch.shape[0] < 2:
                    continue

                # Prepare input as sequence (batch_size, seq_len=1, embedding_dim)
                inputs = batch.unsqueeze(1)
                
                # Target: Product projection target representation
                target_rep = self.product_projection(batch)
                
                # Forward through LSTM & Projection
                output_rep = self(inputs)

                loss = criterion(output_rep, target_rep)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[TRAIN] Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

        # Save trained weights
        torch.save(self.state_dict(), save_checkpoint_path)
        print(f"[INFO] Training complete! Saved checkpoint to '{save_checkpoint_path}'")

    @classmethod
    def load_or_train(
        cls,
        checkpoint_path: str,
        catalog_csv_path: str = "kohler_catalog.csv",
        device: str = "cpu",
        **kwargs
    ):
        """
        Loads the checkpoint if present. Trains from catalog_csv_path and saves to checkpoint_path if missing.
        """
        model = cls(**kwargs)
        model.to(device)

        if os.path.exists(checkpoint_path):
            print(f"[INFO] Loading existing LSTM checkpoint from '{checkpoint_path}'...")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            model.fit_from_catalog(catalog_csv_path=catalog_csv_path, save_checkpoint_path=checkpoint_path)

        model.eval()
        return model
