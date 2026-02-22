#!/usr/bin/env python3
"""
Quick test script to run melody-based inference with imagine_midi_test.mid
"""
import os
import sys
import pickle
import torch
from pathlib import Path
from transformers import BartTokenizer
from models.conbart import Bart
from utils.prosody_utils import getProsody
import miditoolkit

# Set environment
os.environ['PYTHONPATH'] = '.'

# Configuration
MIDI_FILE = "imagine_midi_test.mid"
CHECKPOINT = "bestM2LCkpt.pt"
DICT_FILE = "binary/m2l_dict.pkl"
MODEL_PATH = "facebook/bart-base"
TITLE = "Imagine"
KEYWORDS = ["peace", "unity", "hope", "world"]  # Example keywords for phrases

def get_token_id(tokenizer, token_str):
    """Get token ID from tokenizer, filtering out special tokens"""
    token_ids = tokenizer.encode(token_str)
    bos = set(tokenizer.encode("<s>"))
    eos = set(tokenizer.encode("</s>"))
    special = bos | eos
    for token_id in token_ids:
        if token_id not in special:
            return int(token_id)
    return int(token_ids[-1])

print(f"Loading model and tokenizers...")

# Load dictionary
with open(DICT_FILE, "rb") as f:
    event2word_dict, word2event_dict, lyric2word_dict, _ = pickle.load(f)

# Load tokenizers (using facebook/bart-base as fallback)
src_tknzr = BartTokenizer.from_pretrained(MODEL_PATH)
tgt_tknzr = BartTokenizer.from_pretrained(MODEL_PATH)

# Expand tokenizer vocabulary to match checkpoint
checkpoint_data = torch.load(CHECKPOINT, map_location="cpu")
state_dict = checkpoint_data if isinstance(checkpoint_data, dict) and 'model' not in checkpoint_data else checkpoint_data.get('model', checkpoint_data)

# Get vocab size from checkpoint
vocab_size = None
for key in state_dict.keys():
    if 'embed_tokens.weight' in key:
        vocab_size = state_dict[key].shape[0]
        break

if vocab_size and vocab_size > len(src_tknzr):
    print(f"Expanding tokenizers from {len(src_tknzr)} to {vocab_size} tokens...")
    src_tknzr.add_tokens([f"<extra_id_{i}>" for i in range(vocab_size - len(src_tknzr))])
    tgt_tknzr.add_tokens([f"<extra_id_{i}>" for i in range(vocab_size - len(tgt_tknzr))])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
print("Building model...")
model = Bart(
    event2word_dict=event2word_dict,
    word2event_dict=word2event_dict,
    model_pth=MODEL_PATH,
    src_tknzr=src_tknzr,
    tgt_tknzr=tgt_tknzr,
    hidden_size=768,
    enc_layers=6,
    num_heads=8,
    enc_ffn_kernel_size=2048,
    dropout=0.2,
    cond=True
).to(device)

# Resize model embeddings to match checkpoint
if vocab_size and vocab_size != model.model.model.shared.weight.shape[0]:
    print(f"Resizing model embeddings from {model.model.model.shared.weight.shape[0]} to {vocab_size}...")
    model.model.resize_token_embeddings(vocab_size)

# Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT}...")
try:
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully!")
except Exception as e:
    print(f"Warning: Loading with strict=False due to: {e}")
    model.load_state_dict(state_dict, strict=False)

model.eval()
print("Model loaded successfully!\n")

# Process MIDI file
print(f"Processing MIDI file: {MIDI_FILE}")
midi = miditoolkit.MidiFile(MIDI_FILE)

print(f"  Notes: {len(midi.instruments[0].notes)}")
print(f"  Markers: {len(midi.markers)}")
print(f"  Tempo changes: {len(midi.tempo_changes)}")

# Get prosody
prosody_list = getProsody(MIDI_FILE)
print(f"\nProsody analysis:")
for i, (meter, length) in enumerate(prosody_list[:10]):  # Show first 10
    print(f"  Note {i}: meter={meter}, length={length}")
if len(prosody_list) > 10:
    print(f"  ... and {len(prosody_list) - 10} more")

# Group by phrase markers
group_by_phrase = {}
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

print(f"\nPhrases found: {len(group_by_phrase)}")
for phrase_id, notes in group_by_phrase.items():
    print(f"  Phrase {phrase_id}: {len(notes)} notes")

# Build encoder input
src_words = []
title_str = TITLE.replace('.', '')

# Add title token
title_token = get_token_id(src_tknzr, "<title>")
src_words.append({'meter': title_token, 'length': 0, 'remainder': 0})

# Add title text tokens
title_encoded = src_tknzr.encode(title_str)
for ep in title_encoded:
    if ep in set(src_tknzr.encode("<s>")) | set(src_tknzr.encode("</s>")):
        continue
    src_words.append({'meter': ep, 'length': 0, 'remainder': 0})

# Add prosody template for each phrase
for line_id, line in group_by_phrase.items():
    # Add syllable count token
    line_syllable_num = len(line)
    syll_token = get_token_id(src_tknzr, f"<syllable_{line_syllable_num}>")
    src_words.append({'meter': syll_token, 'length': 0, 'remainder': 0})

    # Add template marker
    template_token = get_token_id(src_tknzr, "<template>")
    src_words.append({'meter': template_token, 'length': 0, 'remainder': 0})

    syllable_count = len(line)
    for word_id, (nid, note, prosody) in enumerate(line):
        # Encode meter label directly with tokenizer (e.g., "<weak>", "<strong>")
        meter_label = prosody[0]
        length_label = prosody[1]

        # Get token IDs
        meter_token = get_token_id(src_tknzr, meter_label)
        length_token = event2word_dict['Length'].get(length_label, 0)
        remainder_token = event2word_dict['Remainder'][f'Remain_{syllable_count - word_id - 1}']

        src_words.append({
            'sentence': line_id,
            'meter': meter_token,
            'length': length_token,
            'remainder': remainder_token,
        })

    # Add period separator
    period_token = get_token_id(src_tknzr, ".")
    src_words.append({'meter': period_token, 'length': 0, 'remainder': 0})

# Add EOS
eos_token = get_token_id(src_tknzr, "</s>")
src_words.append({'meter': eos_token, 'length': 0, 'remainder': 0})

# Prepare decoder input
bos_token = get_token_id(tgt_tknzr, '<s>')
tgt_words = [{
    'word': bos_token,
    'remainder': 0,
}]

# Convert to tensors
enc_inputs = {
    'meter': torch.LongTensor([[w['meter'] for w in src_words]]).to(device),
    'length': torch.LongTensor([[w['length'] for w in src_words]]).to(device),
    'remainder': torch.LongTensor([[w['remainder'] for w in src_words]]).to(device),
}

dec_inputs = {
    'word': torch.LongTensor([[w['word'] for w in tgt_words]]).to(device),
    'remainder': torch.LongTensor([[w['remainder'] for w in tgt_words]]).to(device),
}

# Count total syllables
total_syllables = sum(len(notes) for notes in group_by_phrase.values())

print(f"\nRunning inference...")
print(f"  Temperature: 0.9")
print(f"  Top-k: 50")
print(f"  Total syllables: {total_syllables}")

# Run inference
with torch.no_grad():
    lyrics, ppl = model.infer(
        tgt_tknzr=tgt_tknzr,
        enc_inputs=enc_inputs,
        dec_inputs_gt=dec_inputs,
        sentence_maxlen=1024,
        temperature=0.9,
        topk=50,
        device=device,
        num_syllables=total_syllables
    )

print("\n" + "="*60)
print("GENERATED LYRICS")
print("="*60)
print(lyrics)
print("="*60)
print(f"\nPerplexity: {ppl:.4f}")
