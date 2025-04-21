import datetime
import tqdm
import wandb
import torch
import src.dataset as dataset
import src.evaluate as evaluate
import src.model as model
import src.config as config
import os

torch.manual_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

ds = dataset.Wiki(skip_gram=True)
dl = torch.utils.data.DataLoader(
    dataset=ds, batch_size=config.SKIPGRAM_BATCH_SIZE)

args = (config.VOCAB_SIZE, config.EMBEDDING_DIM)
mFoo = model.SkipGram(*args)
print('mFoo:params', sum(p.numel() for p in mFoo.parameters()))
opFoo = torch.optim.Adam(mFoo.parameters(), lr=config.SKIPGRAM_LR)
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project='mlx7-week1-skipgram', name=f'{ts}')
mFoo.to(dev)

best_loss = float('inf')

for epoch in range(config.SKIPGRAM_EPOCHS):
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
