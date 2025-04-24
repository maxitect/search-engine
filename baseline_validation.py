from engine.data import MSMarcoDataset, collate_fn_marco
from main import Trainer, run_sanity_check
from engine.utils import get_device
import torch
from tqdm import tqdm
from engine.model import TripletLoss

# For each item in validation set just get the GenSim embeddings and then do a dot product with the query embedding
# Compute the accuracy based on that

# Training configs
batch_size = 8
K = 5
# lr = 1e-3
mode = 'random'
# embeddings = 'self-trained'
embeddings = 'word2vec-google-news-300'
num_epochs = 5
# Model configs
D_hidden = 100
D_out = 100
batches_print_frequency = 200
log_to_wandb = True

device = torch.device('cpu')

trainer = Trainer(
    batch_size,
    num_negative_samples=K,
    mode=mode,
    embeddings=embeddings,
    log_to_wandb=log_to_wandb,
    device=device,
    rnn=False,
)

losses = []
accuracies = []
total_num_correct = 0
total_num_samples = 0
loss_fn = TripletLoss()

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(trainer.val_dl)):
        # Zero your gradients for every batch!
        q_vectors, pos_vectors, neg_vectors = batch
        loss = loss_fn(
            q_vectors, pos_vectors,
            neg_vectors.permute(0, 2, 1),
        )

        B, K, D_out = neg_vectors.shape
        accuracy = None
        neg_vectors = neg_vectors.permute(0, 2, 1)

        with torch.no_grad():
            q_pos_sim = torch.cosine_similarity(q_vectors, pos_vectors)
            q_neg_sim = torch.cosine_similarity(
                q_vectors[:, :, None], neg_vectors,
            )
            # Positive sample is at index 0, so check whether the argmax is 0
            # and subsequently count
            num_correct = (
                torch.cat(
                    [q_pos_sim[:, None], q_neg_sim],
                    1,
                ).argmax(axis=1) == 0
            ).sum()
            total_num_correct += num_correct
            total_num_samples += B
        losses.append(loss.item())

print(sum(losses)/len(losses), total_num_correct / total_num_samples)
