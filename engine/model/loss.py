import torch


class TripletLoss():
    def __init__(self, distance_metric: str = 'cosine', margin: float = 1.0):
        self.distance_metric = distance_metric
        if distance_metric == 'cosine':
            self.distance_fn = self.cosine_distance
        else:
            # Use L2 distance
            raise NotImplementedError('L2 distance not implemented')

        self.margin = margin

    def cosine_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return 1 - torch.nn.functional.cosine_similarity(a, b)

    def __call__(
        self,
        query_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the (batched) triplet loss between a query, positive, and
            negative embedding.

        PyTorch torch.nn.functional.triplet_margin_loss requires all three
        embeddings to be the same size, so we reimplement.

        Args:
            query_embedding (torch.Tensor): The query embedding, shape (B, D).
            positive_embedding (torch.Tensor): The positive embedding,
                shape (B, D).
            negative_embedding (torch.Tensor): The negative embedding,
                shape (B, D, M).

        Returns:
            torch.Tensor: The triplet loss, single value.
        """
        B, D = query_embedding.shape
        B_pos, D_pos = positive_embedding.shape
        B_neg, D_neg, M = negative_embedding.shape

        assert B == B_pos == B_neg
        assert D == D_pos == D_neg

        # Reshapes so it is computable.
        # Shape is (B)
        query_positive_distance = self.distance_fn(
            query_embedding, positive_embedding,
        )
        # One distance for each negative example so shape is (B, M)
        query_negative_distance = self.distance_fn(
            query_embedding.view(B, D, 1),
            negative_embedding,
        )
        # Broadcast to (B) using mean to aggregate over negatives and do
        # hinged loss
        loss = torch.maximum(
            query_positive_distance -
            query_negative_distance.mean(dim=1) + self.margin,
            torch.zeros_like(query_positive_distance),
        )
        # Take mean over batch
        loss = loss.mean()
        return loss
