import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_word_embeddings: [B, D_in]
        Returns:
            output_embedding[B, D_out]
        """

        # Pass through the FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output_embedding = x
        return output_embedding

class RNNEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  

        self.rnn = torch.nn.LSTM(
            input_dim, hidden_dim, batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_word_embeddings: [B, D_in]
        Returns:
            output_embedding[B, D_out]
        """ 
        x, _ = self.rnn(x)
        x = self.fc(x)
        output_embedding = x
        return output_embedding

