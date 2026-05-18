from __future__ import annotations

import os
import pickle
import tempfile
import time
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from midi_to_wav import convert
from database import Database

@dataclass
class RuntimeAssets:
    model: Any
    src_tknzr: Any
    tgt_tknzr: Any
    event2word: Dict[str, Dict[str, int]]
    lyric2word: Dict[str, Dict[str, int]]
    device: Any
    torch: Any
    miditoolkit: Any
    prosodic: Any
    get_prosody: Any
    src_keys: List[str]
    tgt_keys: List[str]
    model_path: str
    checkpoint_path: str
    dict_path: str
    load_warning: str
    midi_files: Dict[str, str]
    db_audio_files: Dict[str, str]
    database: Database


SRC_KEYS = ["meter", "length", "remainder"]
TGT_KEYS = ["word", "remainder"]


class ParodyRequest(BaseModel):
    title: str = Field(default="untitled")
    lyrics_lines: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    temperature: float = Field(default=1.3, ge=0.1, le=5.0)
    topk: int = Field(default=3, ge=1, le=100)
    max_tokens: int = Field(default=256, ge=8, le=2048)


app = FastAPI(title="XAI-Lyricist API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.assets = None
app.state.load_error = ""
app.state.lock = Lock()


def _non_empty_lines(lines: List[str]) -> List[str]:
    return [line.strip() for line in lines if line and line.strip()]


def _normalize_keywords(raw_keywords: Optional[List[str]], target_count: int) -> List[str]:
    if target_count <= 0:
        return []
    normalized: List[str] = []
    for item in raw_keywords or []:
        if not item:
            continue
        if "," in item:
            normalized.extend([segment.strip() for segment in item.split(",") if segment.strip()])
        else:
            normalized.append(item.strip())
    if len(normalized) < target_count:
        normalized.extend([""] * (target_count - len(normalized)))
    if len(normalized) > target_count:
        normalized = normalized[:target_count]
    return normalized


def _token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token)
    bos = set(tokenizer.encode("<s>"))
    eos = set(tokenizer.encode("</s>"))
    special = bos | eos
    for token_id in token_ids:
        if token_id not in special:
            return int(token_id)
    return int(token_ids[-1])


def _get_remain_max(lyric2word: Dict[str, Dict[str, int]]) -> int:
    remain_keys = lyric2word.get("Remainder", {})
    max_remain = 0
    for key in remain_keys.keys():
        if isinstance(key, str) and key.startswith("Remain_"):
            try:
                max_remain = max(max_remain, int(key.split("_", 1)[1]))
            except ValueError:
                continue
    return max_remain


def _append_keyword_tokens(src_words: List[Dict[str, int]], keyword: str, assets: RuntimeAssets) -> None:
    if not keyword:
        return
    encoded = assets.src_tknzr.encode(f"<keywords>{keyword.strip()}")
    bos = set(assets.src_tknzr.encode("<s>"))
    eos = set(assets.src_tknzr.encode("</s>"))
    for token_id in encoded:
        if token_id in bos or token_id in eos:
            continue
        src_words.append({"meter": int(token_id), "length": 0, "remainder": 0})


def _build_title_prefix(title: str, assets: RuntimeAssets) -> List[Dict[str, int]]:
    clean_title = (title or "untitled").replace(".", "").strip() or "untitled"
    encoded_prefix = assets.src_tknzr.encode(f"<title>{clean_title}")
    out: List[Dict[str, int]] = []
    for token_id in encoded_prefix:
        if assets.src_tknzr.decode(token_id).strip() == "</s>":
            continue
        out.append({"meter": int(token_id), "length": 0, "remainder": 0})
    return out


def _build_melody_src_words(
    midi_path: str,
    title: str,
    keywords: List[str],
    assets: RuntimeAssets,
) -> Tuple[List[Dict[str, int]], int, int]:
    midi = assets.miditoolkit.MidiFile(midi_path)
    if not midi.instruments or not midi.instruments[0].notes:
        raise ValueError("MIDI must contain at least one instrument with notes.")

    prosody_list = assets.get_prosody(midi_path)
    assert len(prosody_list) == len(midi.instruments[0].notes)

    ## group midi by phrase
    group_by_phrase: Dict[int, List[Tuple[int, Any, Tuple[str, str]]]] = {}
    start = -1
    for idx in range(len(midi.markers)):
        end = midi.markers[idx].time
        if idx not in group_by_phrase:
            group_by_phrase[idx] = []
        for inst in midi.instruments:
            for nid, note in enumerate(inst.notes):
                if note.start > start and note.start <= end:
                    group_by_phrase[idx].append((nid, note, prosody_list[nid]))
        start = midi.markers[idx].time

    # assert len(keywords) == len(group_by_phrase)

    src_words = _build_title_prefix(title, assets)
    syllable_total = 0
    max_remain = _get_remain_max(assets.lyric2word)

    for _, line in group_by_phrase.items():
        line_syllable_num = len(line)
        syllable_total += line_syllable_num
        syll_token = _token_id(assets.src_tknzr, f"<syllable_{line_syllable_num}>")
        template_token = _token_id(assets.src_tknzr, "<template>")
        period_token = _token_id(assets.src_tknzr, ".")

        src_words.append({"meter": syll_token, "length": 0, "remainder": 0})
        src_words.append({"meter": template_token, "length": 0, "remainder": 0})

        remain = line_syllable_num
        for note in line:
            remain -= 1
            mtype, length = note[2][0], note[2][1]
            meter_token = _token_id(assets.src_tknzr, mtype)
            length_id = assets.event2word.get("Length", {}).get(length, 0)
            remain_key = f"Remain_{max(0, min(remain, max_remain))}"
            remain_id = assets.lyric2word.get("Remainder", {}).get(remain_key, 0)
            src_words.append({"meter": meter_token, "length": int(length_id), "remainder": int(remain_id)})

        src_words.append({"meter": period_token, "length": 0, "remainder": 0})

    eos_id = _token_id(assets.src_tknzr, "</s>")
    src_words.append({"meter": eos_id, "length": 0, "remainder": 0})
    return src_words, len(group_by_phrase), syllable_total


def _build_parody_src_words(
    title: str,
    lyrics_lines: List[str],
    keywords: List[str],
    assets: RuntimeAssets,
) -> Tuple[List[Dict[str, int]], int]:
    lines = _non_empty_lines(lyrics_lines)
    if not lines:
        raise ValueError("`lyrics_lines` must include at least one non-empty line.")

    normalized_keywords = _normalize_keywords(keywords, len(lines))
    max_remain = _get_remain_max(assets.lyric2word)
    src_words = _build_title_prefix(title, assets)
    syllable_total = 0

    for line_idx, raw_line in enumerate(lines):
        text = assets.prosodic.Text(raw_line.strip())
        parsed_lines = list(text.lines())
        if not parsed_lines:
            continue
        line = parsed_lines[0]
        syll_attr = getattr(line, "syllables", None)
        syllables = list(syll_attr() if callable(syll_attr) else (syll_attr or []))
        line_syllable_num = len(syllables)
        if line_syllable_num <= 0:
            continue

        syllable_total += line_syllable_num
        syll_token = _token_id(assets.src_tknzr, f"<syllable_{line_syllable_num}>")
        template_token = _token_id(assets.src_tknzr, "<template>")
        period_token = _token_id(assets.src_tknzr, ".")

        src_words.append({"meter": syll_token, "length": 0, "remainder": 0})
        src_words.append({"meter": template_token, "length": 0, "remainder": 0})

        remain = line_syllable_num
        for syllable in syllables:
            remain -= 1
            syllable_text = str(syllable)
            if "'" in syllable_text:
                meter_label = "<strong>"
            elif "`" in syllable_text:
                meter_label = "<substrong>"
            else:
                meter_label = "<weak>"
            length_label = "Long" if "ː" in syllable_text else "Short"
            meter_token = _token_id(assets.src_tknzr, meter_label)
            length_id = assets.event2word.get("Length", {}).get(length_label, 0)
            remain_key = f"Remain_{max(0, min(remain, max_remain))}"
            remain_id = assets.lyric2word.get("Remainder", {}).get(remain_key, 0)
            src_words.append({"meter": meter_token, "length": int(length_id), "remainder": int(remain_id)})

        # _append_keyword_tokens(src_words, normalized_keywords[line_idx], assets)
        src_words.append({"meter": period_token, "length": 0, "remainder": 0})

    if syllable_total <= 0:
        raise ValueError("Failed to extract syllables from the provided lyrics.")

    eos_id = _token_id(assets.src_tknzr, "</s>")
    src_words.append({"meter": eos_id, "length": 0, "remainder": 0})
    return src_words, syllable_total


def _build_model_inputs(src_words: List[Dict[str, int]], assets: RuntimeAssets) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch = assets.torch
    device = assets.device
    src_tensor_data: Dict[str, Any] = {}
    for key in assets.src_keys:
        src_tensor_data[key] = torch.LongTensor([[word[key] for word in src_words]])
    enc_inputs = {key: src_tensor_data[key].to(device) for key in assets.src_keys}

    bos_ids = assets.tgt_tknzr.encode("<s>")
    bos_token = int(bos_ids[1] if len(bos_ids) > 1 else bos_ids[0])
    dec_inputs = {
        "word": torch.LongTensor([[bos_token]]).to(device),
        "remainder": torch.LongTensor([[0]]).to(device),
    }
    return enc_inputs, dec_inputs


def _generate(
    enc_inputs: Dict[str, Any],
    dec_inputs: Dict[str, Any],
    temperature: float,
    topk: int,
    max_tokens: int,
    syllable_total: int,
    assets: RuntimeAssets,
) -> Tuple[str, List[str], float]:
    with app.state.lock:
        with assets.torch.no_grad():
            output, ppl = assets.model.infer(
                tgt_tknzr=assets.tgt_tknzr,
                enc_inputs=enc_inputs,
                dec_inputs_gt=dec_inputs,
                sentence_maxlen=max_tokens,
                temperature=temperature,
                topk=topk,
                device=assets.device,
                num_syllables=max(1, syllable_total),
            )
    lyrics_text = output.replace("<s>", "").replace("</s>", "").strip()
    lyrics_lines = [line.strip() for line in lyrics_text.split(".") if line.strip()]
    return lyrics_text, lyrics_lines, float(ppl)


def _is_likely_local_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("./")
        or value.startswith("../")
        or value.startswith("~")
        or value.startswith("\\\\")
    )


def _pick_pretrained_source(candidates: List[Optional[str]], fallback: str) -> str:
    for candidate in candidates:
        if not candidate or not str(candidate).strip():
            continue
        text = str(candidate).strip()
        expanded = str(Path(text).expanduser())
        if Path(expanded).exists():
            return expanded
        if _is_likely_local_path(text):
            continue
        return text

    fallback_text = str(fallback).strip()
    fallback_expanded = str(Path(fallback_text).expanduser())
    if Path(fallback_expanded).exists():
        return fallback_expanded
    return fallback_text


def _pick_existing_path(candidates: List[Optional[str]], fallback: str) -> str:
    for candidate in candidates:
        if not candidate or not str(candidate).strip():
            continue
        expanded = str(Path(str(candidate).strip()).expanduser())
        if Path(expanded).exists():
            return expanded
    return str(Path(fallback).expanduser())


def _extract_state_dict(loaded_obj: Any) -> Any:
    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj:
            return loaded_obj["state_dict"]
        if "model_state_dict" in loaded_obj:
            return loaded_obj["model_state_dict"]
    return loaded_obj


def _get_checkpoint_vocab_size(state_dict: Any) -> Optional[int]:
    if not isinstance(state_dict, dict):
        return None
    for key in [
        "model.model.shared.weight",
        "model.shared.weight",
        "shared.weight",
        "model.lm_head.weight",
        "lm_head.weight",
    ]:
        value = state_dict.get(key)
        if value is not None and getattr(value, "shape", None):
            return int(value.shape[0])
    return None


def _preferred_added_tokens() -> List[str]:
    tokens = [f"<syllable_{idx}>" for idx in range(50)]
    tokens.extend(["<keywords>", "<title>", "<sep>", "<template>", "<strong>", "<substrong>", "<weak>"])
    return tokens


def _expand_tokenizer_to_vocab(tokenizer: Any, target_vocab: Optional[int]) -> int:
    if target_vocab is None:
        return 0
    added = 0
    if len(tokenizer) < target_vocab:
        primary = [token for token in _preferred_added_tokens() if token not in tokenizer.get_vocab()]
        if primary:
            added += int(tokenizer.add_special_tokens({"additional_special_tokens": primary}))
    while len(tokenizer) < target_vocab:
        pad_token = f"<extra_token_{len(tokenizer)}>"
        added += int(tokenizer.add_special_tokens({"additional_special_tokens": [pad_token]}))
        if added > 10000:
            break
    return added


def _resize_model_vocab_if_needed(model: Any, target_vocab: Optional[int]) -> None:
    if target_vocab is None:
        return
    current = int(model.model.get_input_embeddings().weight.shape[0])
    if current == target_vocab:
        return
    model.model.resize_token_embeddings(target_vocab)
    model.src_word_emb = model.model.get_encoder().embed_tokens
    model.mel_embed.meter_emb = model.src_word_emb
    model.tgt_word_emb = model.model.get_decoder().embed_tokens
    model.lyr_embed.tgt_word_emb = model.tgt_word_emb


def _load_assets() -> RuntimeAssets:
    import torch
    from transformers import BartTokenizer

    from models.conbart import Bart
    from utils.hparams import set_hparams
    from utils.prosody_utils import getProsody

    import miditoolkit
    import prosodic

    load_dotenv()  

    config_path = os.getenv("XAI_CONFIG_PATH", "configs/configs.yaml")
    cfg: Dict[str, Any] = {}
    if Path(config_path).exists():
        cfg = set_hparams(config=config_path)

    model_path = _pick_pretrained_source(
        [os.getenv("XAI_MODEL_PATH"), cfg.get("custom_model_dir"), cfg.get("model_dir")],
        "facebook/bart-base",
    )
    enc_tokenizer_path = _pick_pretrained_source(
        [
            os.getenv("XAI_ENC_TOKENIZER_DIR"),
            cfg.get("enc_tknzr_dir"),
            cfg.get("custom_model_dir"),
            cfg.get("dec_tknzr_dir"),
        ],
        model_path,
    )
    dec_tokenizer_path = _pick_pretrained_source(
        [os.getenv("XAI_DEC_TOKENIZER_DIR"), cfg.get("dec_tknzr_dir"), cfg.get("model_dir"), model_path],
        "facebook/bart-base",
    )
    checkpoint_path = _pick_existing_path([os.getenv("XAI_CHECKPOINT_PATH"), "bestM2LCkpt.pt"], "bestM2LCkpt.pt")
    dict_path = _pick_existing_path([os.getenv("XAI_DICT_PATH"), "binary/m2l_dict.pkl"], "binary/m2l_dict.pkl")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(dict_path).exists():
        raise FileNotFoundError(f"Dictionary not found: {dict_path}")

    loaded_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(loaded_obj)
    checkpoint_vocab_size = _get_checkpoint_vocab_size(state_dict)

    with open(dict_path, "rb") as handle:
        event2word_dict, word2event_dict, lyric2word_dict, _ = pickle.load(handle)
    if "Remainder" not in lyric2word_dict and "Remainder" in event2word_dict:
        lyric2word_dict = event2word_dict

    src_tknzr = BartTokenizer.from_pretrained(enc_tokenizer_path)
    tgt_tknzr = BartTokenizer.from_pretrained(dec_tokenizer_path)
    _expand_tokenizer_to_vocab(src_tknzr, checkpoint_vocab_size)
    _expand_tokenizer_to_vocab(tgt_tknzr, checkpoint_vocab_size)

    requested_device = os.getenv("XAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    model = Bart(
        event2word_dict=event2word_dict,
        word2event_dict=word2event_dict,
        model_pth=model_path,
        src_tknzr=src_tknzr,
        tgt_tknzr=tgt_tknzr,
        hidden_size=int(cfg.get("hidden_size", 768)),
        enc_layers=int(cfg.get("n_layers", 6)),
        num_heads=int(cfg.get("n_head", 8)),
        enc_ffn_kernel_size=int(cfg.get("ffn_hidden", 2048)),
        dropout=float(cfg.get("drop_prob", 0.2)),
        cond=bool(cfg.get("cond", True)),
    ).to(device)
    _resize_model_vocab_if_needed(model, checkpoint_vocab_size)

    load_warning = ""
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as strict_exc:
        model.load_state_dict(state_dict, strict=False)
        load_warning = f"Loaded checkpoint with strict=False: {strict_exc}"
    model.eval()

    midi_path = os.getenv("MIDI_FILES_PATH", "midi_files")
    midi_files = {
        "find my way back home": os.path.join(midi_path, "find_my_way_back_home.mid"),
        "imagine": os.path.join(midi_path, "imagine.mid"),
        "million reasons": os.path.join(midi_path, "million_reasons.mid"),
        "set fire to the rain": os.path.join(midi_path, "set_fire_to_the_rain.mid"),
        "stay with me": os.path.join(midi_path, "stay_with_me.mid")
    } # map it to the mid file, hardcoded to prevent attacks

    db_audio_files = {
        "find my way back home": "find my way back home.wav",
        "imagine": "imagine.mp3",
        "million reasons": "million reasons.wav",
        "set fire to the rain": "set fire to the rain.wav",
        "stay with me": "stay with me.wav"
    }

    database = Database(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    return RuntimeAssets(
        model=model,
        src_tknzr=src_tknzr,
        tgt_tknzr=tgt_tknzr,
        event2word=event2word_dict,
        lyric2word=lyric2word_dict,
        device=device,
        torch=torch,
        miditoolkit=miditoolkit,
        prosodic=prosodic,
        get_prosody=getProsody,
        src_keys=list(SRC_KEYS),
        tgt_keys=list(TGT_KEYS),
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        dict_path=dict_path,
        load_warning=load_warning,
        midi_files=midi_files,
        db_audio_files=db_audio_files,
        database=database,
    )


def _ensure_assets() -> RuntimeAssets:
    if app.state.assets is not None:
        return app.state.assets
    try:
        app.state.assets = _load_assets()
        app.state.load_error = ""
    except Exception as exc:
        app.state.load_error = str(exc)
        raise HTTPException(status_code=503, detail=f"Model is not ready: {exc}") from exc
    return app.state.assets


@app.on_event("startup")
def on_startup() -> None:
    try:
        app.state.assets = _load_assets()
        app.state.load_error = ""
    except Exception as exc:
        app.state.assets = None
        app.state.load_error = str(exc)
        raise RuntimeError(f"Model load failed during startup: {exc}") from exc


@app.get("/health")
def health() -> Dict[str, Any]:
    ready = app.state.assets is not None
    response = {
        "ok": ready,
        "service": "xai-lyricist-api",
        "model_loaded": ready,
        "load_error": app.state.load_error,
    }
    if ready:
        assets: RuntimeAssets = app.state.assets
        response["device"] = str(assets.device)
        response["model_path"] = assets.model_path
        response["checkpoint_path"] = assets.checkpoint_path
        response["dict_path"] = assets.dict_path
        response["load_warning"] = assets.load_warning
    return response


@app.post("/generate/melody")
async def generate_melody(
    request: Request,
    file_name: str = Form(""),
    title: str = Form("untitled"),
    keywords: Optional[List[str]] = Form(None),
    temperature: float = Form(1.2),
    topk: int = Form(3),
    max_tokens: int = Form(512),
) -> Dict[str, Any]:
    assets = _ensure_assets()

    if request.headers.get("Authorization") == None or "Bearer " not in request.headers.get("Authorization"):
        raise HTTPException(status_code=400, detail="auth token is required")
    
    token = request.headers.get("Authorization").replace("Bearer ", "")

    if not assets.database.validate_user(token):
        raise HTTPException(status_code=400, detail="valid user auth token is required")

    if file_name == "":
        raise HTTPException(status_code=400, detail="`midi_file` is required.")

    if file_name not in assets.midi_files:
        raise HTTPException(status_code=400, detail="midi file is not found")


    start_time = time.time()
    tmp_path = assets.midi_files[file_name]
    try:
        src_words, phrase_count, _ = _build_melody_src_words(
            midi_path=tmp_path,
            title=title,
            keywords=keywords or [],
            assets=assets,
        )
        enc_inputs, dec_inputs = _build_model_inputs(src_words, assets)
        lyrics_text, lyrics_lines, ppl = _generate(
            enc_inputs=enc_inputs,
            dec_inputs=dec_inputs,
            temperature=float(temperature),
            topk=int(topk),
            max_tokens=int(max_tokens),
            syllable_total=30,
            assets=assets,
        )

        wav_data = convert(tmp_path, lyrics_text)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Melody generation failed: {exc}") from exc

    assets.database.insert_song(wav_data, token, title, keywords, lyrics_text, [], {})
    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "mode": "melody",
        "title": title,
        "lyrics_text": lyrics_text,
        "lyrics_lines": lyrics_lines,
        "meta": {
            "phrase_count": phrase_count,
            "temperature": float(temperature),
            "topk": int(topk),
            "max_tokens": int(max_tokens),
            "device": str(assets.device),
            "elapsed_ms": elapsed_ms,
            "perplexity": float(ppl),
        },
    }


@app.post("/generate/parody")
def generate_parody(request: ParodyRequest) -> Dict[str, Any]:
    assets = _ensure_assets()
    lines = _non_empty_lines(request.lyrics_lines)
    if not lines:
        raise HTTPException(status_code=400, detail="`lyrics_lines` must include at least one non-empty line.")

    start_time = time.time()
    try:
        src_words, syllable_total = _build_parody_src_words(
            title=request.title,
            lyrics_lines=lines,
            keywords=request.keywords,
            assets=assets,
        )
        enc_inputs, dec_inputs = _build_model_inputs(src_words, assets)
        lyrics_text, lyrics_lines, ppl = _generate(
            enc_inputs=enc_inputs,
            dec_inputs=dec_inputs,
            temperature=float(request.temperature),
            topk=int(request.topk),
            max_tokens=int(request.max_tokens),
            syllable_total=syllable_total,
            assets=assets,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parody generation failed: {exc}") from exc

    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "mode": "parody",
        "title": request.title,
        "lyrics_text": lyrics_text,
        "lyrics_lines": lyrics_lines,
        "meta": {
            "input_line_count": len(lines),
            "temperature": float(request.temperature),
            "topk": int(request.topk),
            "max_tokens": int(request.max_tokens),
            "device": str(assets.device),
            "elapsed_ms": elapsed_ms,
            "perplexity": float(ppl),
        },
    }
