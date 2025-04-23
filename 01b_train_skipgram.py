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
from torch.utils.data import random_split

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
    args = parser.parse_args()

    torch.manual_seed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    # Load full dataset
    full_ds = dataset.Wiki(skip_gram=True)

    # Split dataset into train, validation, and test
    dataset_size = len(full_ds)
    test_size = int(dataset_size * config.TEST_SPLIT)
    val_size = int(dataset_size * config.VALIDATION_SPLIT)
    train_size = dataset_size - test_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(
        f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    # Create data loaders
    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=config.SKIPGRAM_BATCH_SIZE, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        dataset=val_ds, batch_size=config.SKIPGRAM_BATCH_SIZE)
    test_dl = torch.utils.data.DataLoader(
        dataset=test_ds, batch_size=config.SKIPGRAM_BATCH_SIZE)

    # Calculate word frequency distribution for negative sampling
    word_freq = np.array(full_ds.word_freqs)
    word_freq = np.power(word_freq, 0.75)  # Raise to 3/4 power as per paper
    # Normalize to get probability distribution
    word_freq = word_freq / np.sum(word_freq)

    # Model instantiation with both input and output embeddings
    voc_size = config.VOCAB_SIZE
    emb_dim = config.EMBEDDING_DIM

    model_args = (voc_size, emb_dim)
    model = SkipGram(*model_args)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    num_neg_samples = config.NEGATIVE_SAMPLES

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
        model.load_state_dict(checkpoint)

        try:
            epoch_from_filename = int(pth_files[0].split('.')[0])
            start_epoch = epoch_from_filename
            global_step = start_epoch * len(train_dl)
            print(f"Resuming from epoch {start_epoch}, step {global_step}")
        except ValueError:
            print(
                "Couldn't determine epoch from filename, starting from epoch 0"
            )

    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SKIPGRAM_LR)

    total_steps = config.SKIPGRAM_EPOCHS * len(train_dl)
    scheduler = get_lr_scheduler(
        optimizer,
        config.SKIPGRAM_LR_SCHEDULE,
        warmup_steps=config.SKIPGRAM_WARMUP_STEPS,
        total_steps=total_steps
    )

    run_name = f'skipgram_{ts}_resumed' if args.resume else f'skipgram_{ts}'
    wandb.init(project='search-engine', name=run_name)

    # Record key configuration parameters
    wandb.config.update({
        "neg_samples": num_neg_samples,
        "embedding_dim": emb_dim,
        "vocab_size": voc_size,
        "batch_size": config.SKIPGRAM_BATCH_SIZE,
        "lr": config.SKIPGRAM_LR,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size
    })

    for epoch in range(start_epoch, config.SKIPGRAM_EPOCHS):
        # Training phase
        model.train()
        epoch_train_loss = 0
        batch_count = 0
        prgs = tqdm.tqdm(train_dl, desc=f'Epoch {epoch+1}', leave=False)

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

            optimizer.zero_grad()
            # Get embeddings for input, positive and negative samples
            input_embeds, pos_embeds, neg_embeds = model(ipt, trg, neg_samples)

            # Calculate loss using negative sampling
            loss = negative_sampling_loss(input_embeds, pos_embeds, neg_embeds)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({'learning_rate': current_lr, 'step': global_step})

            epoch_train_loss += loss.item()
            batch_count += 1
            global_step += 1

            wandb.log({'train_loss': loss.item(), 'step': global_step})
            prgs.set_postfix({'loss': loss.item()})

            if step % 10_000 == 0:
                evaluate.topk(model)

        avg_train_loss = epoch_train_loss / batch_count

        # Validation phase
        val_loss = evaluate.evaluate_model(model, word_freq, val_dl, "val")
        test_loss = evaluate.evaluate_model(model, word_freq, test_dl, "test")

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'epoch_train_loss': avg_train_loss,
            'epoch_val_loss': val_loss,
            'epoch_test_loss': test_loss
        })

        # Checkpoint saving
        checkpoint_name = f'{epoch + 1}.pth'
        checkpoint_path = os.path.join(
            config.SKIPGRAM_CHECKPOINT_DIR, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.SKIPGRAM_BEST_MODEL_PATH)
            print(
                f'New best model saved at epoch {epoch+1} '
                f'with validation loss {best_val_loss:.4f}'
            )

        artifact = wandb.Artifact('model-weights', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    # Final evaluation on test set
    test_loss = evaluate.evaluate_model(model, word_freq, test_dl, "test")
    wandb.log({'final_test_loss': test_loss})

    wandb.finish()
    print(
        f'Training completed. Best validation loss: {best_val_loss:.4f}, '
        f'Final test loss: {test_loss:.4f}')


if __name__ == "__main__":
    main()
