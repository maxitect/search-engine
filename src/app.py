import os
import json
import torch
import pickle
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import src.config as config
import src.model as model

os.makedirs(config.LOG_DIR_PATH, exist_ok=True)
if not os.path.exists(config.LOG_PATH):
    with open(config.LOG_PATH, 'w') as f:
        f.write('')

app = FastAPI()


class Post(BaseModel):
    author: str
    title: str
    timestamp: str


vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))
skipgram = model.SkipGram(len(vocab_to_int), config.EMBEDDING_DIM)
skipgram.load_state_dict(torch.load(config.SKIPGRAM_BEST_MODEL_PATH))
skipgram.eval()

regressor = model.Regressor(emb_dim=config.EMBEDDING_DIM, domain_size=1000)
regressor.load_state_dict(torch.load(config.REGRESSOR_BEST_MODEL_PATH))
regressor.eval()

domain_to_idx = torch.load(config.DOMAIN_MAPPING_PATH)


def extract_domain(title):
    words = title.lower().split()
    for word in words:
        if '.' in word:
            return word
    return '<no_domain>'


def preprocess_title(title):
    title_tokens = title.lower().split()
    title_ids = []
    for word in title_tokens[:config.MAX_TITLE_LENGTH]:
        if word in vocab_to_int:
            title_ids.append(vocab_to_int[word])
        else:
            title_ids.append(0)

    if len(title_ids) < config.MAX_TITLE_LENGTH:
        title_ids.extend([0] * (config.MAX_TITLE_LENGTH - len(title_ids)))

    return torch.tensor([title_ids])


def get_time_features(timestamp):
    dt = datetime.datetime.fromisoformat(timestamp)
    day_of_week = dt.weekday()
    hour_of_day = dt.hour
    return torch.tensor([[day_of_week/6, hour_of_day/23]], dtype=torch.float32)


@app.get("/ping")
def ping():
    return "ok"


@app.get("/version")
def version():
    return {"version": config.MODEL_VERSION}


@app.get("/logs")
def logs():
    with open(config.LOG_PATH, 'r') as f:
        log_entries = f.readlines()

    parsed_logs = []
    for entry in log_entries:
        if entry.strip():
            try:
                parsed_logs.append(json.loads(entry))
            except json.JSONDecodeError:
                continue

    return {"logs": parsed_logs}


@app.post("/how_many_upvotes")
async def how_many_upvotes(post: Post):
    start_time = datetime.datetime.now().timestamp()

    try:
        title_ids = preprocess_title(post.title)
        domain = extract_domain(post.title)
        domain_idx = torch.tensor([domain_to_idx.get(domain, 0)])
        time_feats = get_time_features(post.timestamp)
        title_length = torch.tensor(
            [min(len(post.title), 200)/200], dtype=torch.float32)

        with torch.no_grad():
            title_emb = skipgram.emb(title_ids).mean(dim=1)
            prediction = regressor(title_emb, domain_idx,
                                   time_feats, title_length)

        upvotes = max(1, int(prediction.item()))

        end_time = datetime.datetime.now().timestamp()
        latency = (end_time - start_time) * 1000

        log_entry = {
            "Latency": latency,
            "Version": config.MODEL_VERSION,
            "Timestamp": end_time,
            "Input": post.model_dump(),
            "Prediction": upvotes
        }

        with open(config.LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        return {"upvotes": upvotes}

    except Exception as e:
        end_time = datetime.datetime.now().timestamp()
        latency = (end_time - start_time) * 1000

        log_entry = {
            "Latency": latency,
            "Version": config.MODEL_VERSION,
            "Timestamp": end_time,
            "Input": post.dict(),
            "Error": str(e)
        }

        with open(config.LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        raise HTTPException(status_code=500, detail=str(e))
