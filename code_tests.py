import torch


def test_triplet_loss():
    from engine.model import TripletLoss

    loss = TripletLoss(margin=1.0)
    N = 4
    M = 5
    D = 3
    print('Random embeddings')
    query_embedding = torch.randn(N, D)
    positive_embedding = torch.randn(N, D)
    negative_embedding = torch.randn(N, D, M)

    print(loss(query_embedding, positive_embedding, negative_embedding))

    print('Same embeddings')
    query_embedding = torch.tensor(
        [0, 2, 0], dtype=torch.float32,
    ).reshape(1, D)
    positive_embedding = torch.tensor(
        [1, 1, 1], dtype=torch.float32,
    ).reshape(1, D)
    negative_embedding = torch.tensor(
        [
            [2, 4, 2],
            [2, 4, 2],
        ], dtype=torch.float32,
    ).reshape(1, D, 2)
    loss = loss(query_embedding, positive_embedding, negative_embedding)
    print(loss)


test_triplet_loss()
