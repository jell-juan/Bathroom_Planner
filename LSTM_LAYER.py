from sentence_transformers import SentenceTransformer
from torch import nn
import torch

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
        """
        Initializes the LSTM Preference Encoder.

        Args:
            embedding_model_name (str): SentenceTransformer model name to convert text to vectors.
            embedding_dim (int): Dimensionality of the sentence embeddings (384 for all-MiniLM-L6-v2).
            hidden_dim (int): The number of features in the LSTM hidden state.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
        """
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
        """
        Converts a list of preference text strings into a tensor of dense embeddings.

        Args:
            preferences (list of str): Ex. ["modern styling", "matte black fixtures"]

        Returns:
            torch.Tensor: Shape (1, seq_len, embedding_dim)
        """
        if not preferences:
            # Return a zero tensor if preferences are empty
            return torch.zeros((1, 1, self.embedding_dim))

        # Generate sentence embeddings
        embeddings_np = self.embedding_model.encode(preferences)
        embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)

        # Reshape to (batch_size=1, sequence_length, embedding_dim)
        return embeddings_tensor.unsqueeze(0)

    def forward(self, preference_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Feeds the sequence of embeddings through the LSTM to produce a single aggregated preference vector.

        Args:
            preference_embeddings (torch.Tensor): Shape (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Output representation from the final step of the LSTM.
                          Shape: (batch_size, hidden_dim * 2 if bidirectional else hidden_dim)
        """
        # LSTM output: (batch, seq_len, num_directions * hidden_dim)
        # hn: (num_layers * num_directions, batch, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(preference_embeddings)

        # Retrieve the final sequential output representing the entire preference context
        final_vector = lstm_out[:, -1, :]
        final_vector = self.projection(final_vector)

        return final_vector