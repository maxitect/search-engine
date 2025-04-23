import datetime
import tqdm
import wandb
import torch
import src.dataset as dataset
import src.evaluate as evaluate
from src.models.skipgram import SkipGram, negative_sampling_loss
import src.config as config
import os
import argparse
import numpy as np

from src.utils.lr_scheduler import get_lr_scheduler


def main():
    parser = argparse.ArgumentParser(description='Train SkipGram model')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--artifact_version',
        type=str,
        default=None,
        help='Wandb artifact version to resume from (e.g., v4)'
    )
    parser.add_argument(
        '--neg_samples',
        type=int,
        default=15,
        help='Number of negative samples per positive sample'
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    ds = dataset.Wiki(skip_gram=True)
    dl = torch.utils.data.DataLoader(
        dataset=ds, batch_size=config.SKIPGRAM_BATCH_SIZE)

    # Calculate word frequency distribution for negative sampling
    word_freq = np.array(ds.word_freqs)
    word_freq = np.power(word_freq, 0.75)  # Raise to 3/4 power as per paper
    # Normalize to get probability distribution
    word_freq = word_freq / np.sum(word_freq)

    # Model instantiation with both input and output embeddings
    voc_size = config.VOCAB_SIZE
    emb_dim = config.EMBEDDING_DIM

    model_args = (voc_size, emb_dim)
    mFoo = SkipGram(*model_args)
    print('mFoo:params', sum(p.numel() for p in mFoo.parameters()))

    start_epoch = 0
    best_loss = float('inf')
    global_step = 0
    num_neg_samples = args.neg_samples

    if args.resume and args.artifact_version:
        print(f"Resuming training from artifact {args.artifact_version}")

        api = wandb.Api()
        artifact = api.artifact(
            'maxime-downe-founders-and-coders/mlx7-week1-skipgram/'
            f'model-weights:{args.artifact_version}')
        artifact_dir = artifact.download()

        pth_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
        if not pth_files:
            raise ValueError("No .pth files found in the artifact")

        model_path = os.path.join(artifact_dir, pth_files[0])
        checkpoint = torch.load(model_path)
        mFoo.load_state_dict(checkpoint)

        try:
            epoch_from_filename = int(pth_files[0].split('.')[0])
            start_epoch = epoch_from_filename
            global_step = start_epoch * len(dl)
            print(f"Resuming from epoch {start_epoch}, step {global_step}")
        except ValueError:
            print(
                "Couldn't determine epoch from filename, starting from epoch 0"
            )

    mFoo.to(dev)
    opFoo = torch.optim.Adam(mFoo.parameters(), lr=config.SKIPGRAM_LR)

    total_steps = config.SKIPGRAM_EPOCHS * len(dl)
    scheduler = get_lr_scheduler(
        opFoo,
        config.SKIPGRAM_LR_SCHEDULE,
        warmup_steps=config.SKIPGRAM_WARMUP_STEPS,
        total_steps=total_steps
    )

    run_name = f'skipgram_{ts}_resumed' if args.resume else f'skipgram_{ts}'
    wandb.init(project='search-engine', name=run_name)

    for epoch in range(start_epoch, config.SKIPGRAM_EPOCHS):
        epoch_loss = 0
        batch_count = 0
        prgs = tqdm.tqdm(dl, desc=f'Epoch {epoch+1}', leave=False)
        for step, (ipt, trg) in enumerate(prgs):
            ipt, trg = ipt.to(dev), trg.to(dev)
            batch_size = ipt.size(0)

            # Sample negative words based on
            # unigram distribution raised to 3/4 power
            neg_samples = torch.multinomial(
                torch.tensor(word_freq, device=dev),
                batch_size * num_neg_samples,
                replacement=True
            ).view(batch_size, num_neg_samples)

            opFoo.zero_grad()
            # Get embeddings for input, positive and negative samples
            input_embeds, pos_embeds, neg_embeds = mFoo(ipt, trg, neg_samples)

            # Calculate loss using negative sampling
            loss = negative_sampling_loss(input_embeds, pos_embeds, neg_embeds)
            loss.backward()
            opFoo.step()

            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({'learning_rate': current_lr})

            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1

            wandb.log({'loss': loss.item(), 'step': global_step})
            if step % 10_000 == 0:
                evaluate.topk(mFoo)

        avg_epoch_loss = epoch_loss / batch_count
        wandb.log({'epoch_loss': avg_epoch_loss, 'epoch': epoch})

        checkpoint_name = f'{epoch + 1}.pth'
        checkpoint_path = os.path.join(
            config.SKIPGRAM_CHECKPOINT_DIR, checkpoint_name)
        torch.save(mFoo.state_dict(), checkpoint_path)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(mFoo.state_dict(), config.SKIPGRAM_BEST_MODEL_PATH)
            print(
                f'New best model saved at epoch {epoch+1} '
                f'with loss {best_loss:.4f}'
            )

        artifact = wandb.Artifact('model-weights', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    wandb.finish()
    print(f'Training completed. Best model saved with loss {best_loss:.4f}')


if __name__ == "__main__":
    main()
