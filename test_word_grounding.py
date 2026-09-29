"""
Unit test and verification for Word-Level Visual Grounding & Subword Aggregation in BLIP.
Verifies:
1. Subword token grouping: e.g., 'croche' -> ['cr', '##oche']
2. Single token words: e.g., 'wearing' -> ['wearing']
3. Position distinction for repeated words: e.g., 1st 'a' vs 2nd 'a' vs 3rd 'a'
4. Cross-attention aggregation: mean(A_subwords) over 576 visual patches
5. Pairwise dissimilarity among content words: man, wearing, croche, hat, standing, room.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("=" * 80)
print("TEST: WORD-LEVEL VISUAL GROUNDING & SUBWORD TOKEN AGGREGATION")
print("=" * 80)

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
model.eval()

# Sample image
sample_images = list(Path('data/flickr8k/Images').glob('*.jpg'))
img_path = sample_images[0] if sample_images else None
if img_path:
    img = Image.open(img_path).convert('RGB')
else:
    img = Image.new('RGB', (384, 384), color=(128, 128, 128))

test_caption = "a man wearing a croche hat while standing in a room"
print(f"Input Image   : {img_path.name if img_path else 'Sample'}")
print(f"Target Caption: \"{test_caption}\"")

# 1. Tokenize & Group Subwords into Words
encoding = processor.tokenizer(test_caption, return_tensors="pt")
input_ids = encoding.input_ids[0].tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

print("\n--- RAW TOKENS FROM TOKENIZER ---")
for i, (t_id, tok) in enumerate(zip(input_ids, tokens)):
    print(f"  Pos [{i:2d}] Token ID: {t_id:<6} | Token: \"{tok}\"")

words_data = []
current_word_tokens = []
current_word_ids = []
current_word_indices = []
current_word_str = ""

for idx, (tok_id, tok) in enumerate(zip(input_ids, tokens)):
    if tok in ("[CLS]", "[SEP]", "[PAD]"):
        continue
    if tok.startswith("##"):
        current_word_tokens.append(tok)
        current_word_ids.append(tok_id)
        current_word_indices.append(idx)
        current_word_str += tok[2:]
    else:
        if current_word_tokens:
            words_data.append({
                "word_index": len(words_data) + 1,
                "word": current_word_str,
                "tokens": current_word_tokens,
                "token_ids": current_word_ids,
                "token_indices": current_word_indices,
                "is_multi_token": len(current_word_tokens) > 1
            })
        current_word_tokens = [tok]
        current_word_ids = [tok_id]
        current_word_indices = [idx]
        current_word_str = tok

if current_word_tokens:
    words_data.append({
        "word_index": len(words_data) + 1,
        "word": current_word_str,
        "tokens": current_word_tokens,
        "token_ids": current_word_ids,
        "token_indices": current_word_indices,
        "is_multi_token": len(current_word_tokens) > 1
    })

print("\n--- RECONSTRUCTED WORDS & SUBWORD MAPPINGS ---")
for w in words_data:
    multi_tag = f"(Multi-Token Subwords: {w['tokens']})" if w['is_multi_token'] else "(Single Token)"
    print(f"Word #{w['word_index']:2d}: \"{w['word']:<10}\" | Positions: {w['token_indices']} | IDs: {w['token_ids']} | {multi_tag}")

# 2. Extract Cross-Attention Matrix
inputs = processor(images=img, return_tensors="pt")
pixel_values = inputs.pixel_values

with torch.no_grad():
    vision_out = model.vision_model(pixel_values=pixel_values)
    image_embeds = vision_out[0]
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
    
    decoder_out = model.text_decoder(
        input_ids=torch.tensor([input_ids]),
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        output_attentions=True,
        return_dict=True
    )

cross_attentions = decoder_out.cross_attentions
# Layers 8 to 11
top_layers = torch.stack(cross_attentions[-4:], dim=0)
# Shape: [seq_len, 576] (dropping CLS token at index 0)
token_attn_matrix = top_layers.mean(dim=(0, 1, 2)).squeeze(0)[:, 1:]

# 3. Aggregate Subword Attentions for Each Word
seq_mean = token_attn_matrix.mean(dim=0, keepdim=True)
seq_std = token_attn_matrix.std(dim=0, keepdim=True) + 1e-6
z_relevance = torch.clamp((token_attn_matrix - seq_mean) / seq_std, min=0.0)

for w in words_data:
    indices = w["token_indices"]
    # Mean of attention across subword tokens of the word
    if len(indices) == 1:
        w_raw_attn = token_attn_matrix[indices[0]].numpy()
        w_relevance = z_relevance[indices[0]].numpy()
    else:
        w_raw_attn = token_attn_matrix[indices].mean(dim=0).numpy()
        w_relevance = z_relevance[indices].mean(dim=0).numpy()
        
    w["raw_attn"] = w_raw_attn
    w["relevance"] = w_relevance
    w["peak_patch"] = int(w_relevance.argmax())

# 4. Verify Repeated Words and Word-Level Peak Statistics
print("\n--- WORD VISUAL GROUNDING PEAK PATCHES ---")
for w in words_data:
    p = w["peak_patch"]
    print(f"Word #{w['word_index']:2d} \"{w['word']:<10}\": Peak Patch #{p:3d} (Row {p//24:2d}, Col {p%24:2d}) | Tokens: {w['tokens']}")

# 5. Check Dissimilarity Between Target Content Words
target_words = ["man", "wearing", "croche", "hat", "standing", "room"]
selected_word_objs = [w for w in words_data if w["word"] in target_words]

print("\n--- DISSIMILARITY MATRIX BETWEEN TARGET CONTENT WORDS ---")
for i in range(len(selected_word_objs)):
    for j in range(i + 1, min(i + 3, len(selected_word_objs))):
        w1 = selected_word_objs[i]
        w2 = selected_word_objs[j]
        v1 = w1["relevance"]
        v2 = w2["relevance"]
        
        mad = float(np.mean(np.abs(v1 - v2)))
        l2 = float(np.linalg.norm(v1 - v2))
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        
        print(f"Diff between \"{w1['word']}\" (pos {w1['token_indices']}) vs \"{w2['word']}\" (pos {w2['token_indices']}):")
        print(f"  • Mean Absolute Difference : {mad:.5f}")
        print(f"  • Euclidean Distance (L2)  : {l2:.4f}")
        print(f"  • Cosine Similarity        : {cos:.4f}")
        print(f"  • Verified Distinct?       : {'YES, DIFFERENT!' if mad > 0.01 else 'IDENTICAL'}")

# 6. Verify Repeated "a" Positions
a_words = [w for w in words_data if w["word"] == "a"]
print(f"\nRepeated word \"a\" count: {len(a_words)}")
for i, w in enumerate(a_words, 1):
    print(f"  Occurrence {i}: Word Index #{w['word_index']}, Token Position: {w['token_indices']}, Peak Patch: #{w['peak_patch']}")
