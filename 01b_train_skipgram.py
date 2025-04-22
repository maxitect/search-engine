import datetime
import tqdm
import wandb
import torch
import src.dataset as dataset
import src.evaluate as evaluate
import src.model as model
import src.config as config
import os
import argparse


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

    ds = dataset.Wiki(skip_gram=True)
    dl = torch.utils.data.DataLoader(
        dataset=ds, batch_size=config.SKIPGRAM_BATCH_SIZE)

    model_args = (config.VOCAB_SIZE, config.EMBEDDING_DIM)
    mFoo = model.SkipGram(*model_args)
    print('mFoo:params', sum(p.numel() for p in mFoo.parameters()))

    start_epoch = 0
    best_loss = float('inf')

    if args.resume and args.artifact_version:
        print(f"Resuming training from artifact {args.artifact_version}")

        api = wandb.Api()
        artifact = api.artifact(
            'maxime-downe/mlx7-week1-skipgram/model-weights:'
            f'{args.artifact_version}'
        )
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
            print(f"Resuming from epoch {start_epoch}")
        except ValueError:
            print(
                "Couldn't determine epoch from filename, starting from epoch 0"
            )

    mFoo.to(dev)
    opFoo = torch.optim.Adam(mFoo.parameters(), lr=config.SKIPGRAM_LR)
    criterion = torch.nn.CrossEntropyLoss()

    run_name = f'{ts}_resumed' if args.resume else ts
    wandb.init(project='mlx7-week1-skipgram', name=run_name)

    for epoch in range(start_epoch, config.SKIPGRAM_EPOCHS):
        epoch_loss = 0
        batch_count = 0
        prgs = tqdm.tqdm(dl, desc=f'Epoch {epoch+1}', leave=False)
        for i, (ipt, trg) in enumerate(prgs):
            ipt, trg = ipt.to(dev), trg.to(dev)
            opFoo.zero_grad()
            out = mFoo(ipt)
            loss = criterion(out, trg.view(-1))
            loss.backward()
            opFoo.step()

            epoch_loss += loss.item()
            batch_count += 1

            wandb.log({'loss': loss.item()})
            if i % 10_000 == 0:
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
