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
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("=" * 70)
print("DEBUGGING BLIP CROSS-ATTENTION FOR VISUAL GROUNDING")
print("=" * 70)

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
model.eval()

# Select an image
img_path = Path('data/flickr8k/Images/1000268201_693b08cb0e.jpg')
img = Image.open(img_path).convert('RGB')
inputs = processor(images=img, return_tensors='pt')
pixel_values = inputs.pixel_values

# 1. Generate Caption & Token IDs
with torch.no_grad():
    output_ids = model.generate(pixel_values=pixel_values, max_length=32)[0]

full_caption = processor.decode(output_ids, skip_special_tokens=True)
print(f"Image             : {img_path.name}")
print(f"Generated Caption : \"{full_caption}\"")
print(f"Total Output IDs  : {len(output_ids)}")

# 2. Vision Encoder Forward Pass
with torch.no_grad():
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    image_embeds = vision_outputs[0] # [1, 577, 768]
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
    
    # 3. Text Decoder Forward Pass with Cross-Attention output
    dec_input_ids = output_ids.unsqueeze(0)
    decoder_outputs = model.text_decoder(
        input_ids=dec_input_ids,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        output_attentions=True,
        return_dict=True
    )

cross_attentions = decoder_outputs.cross_attentions
print(f"Cross-Attention Layers Count : {len(cross_attentions)} (one per Decoder layer)")
print(f"Each Layer Cross-Attn Shape  : {list(cross_attentions[0].shape)} -> [batch=1, heads=12, text_len={len(output_ids)}, vision_tokens=577]")

# Layer 11 (last semantic layer)
last_layer = cross_attentions[-1][0] # [12 heads, text_len, 577]
# Mean across heads
token_attns = last_layer.mean(dim=0) # [text_len, 577]

print("\n--- TOKEN-BY-TOKEN ATTENTION BREAKDOWN ---")
tokens_info = []
for idx, token_id in enumerate(output_ids):
    token_str = processor.decode([token_id]).strip()
    attn_vec = token_attns[idx, 1:] # 576 spatial patches
    min_v = attn_vec.min().item()
    max_v = attn_vec.max().item()
    mean_v = attn_vec.mean().item()
    std_v = attn_vec.std().item()
    
    tokens_info.append({
        "idx": idx,
        "token_id": token_id.item(),
        "token_str": token_str,
        "attn_vector": attn_vec
    })
    
    print(f"[{idx:2d}] Token ID={token_id.item():<6} | Token: \"{token_str:<10}\" | Shape: {list(attn_vec.shape)} | Min: {min_v:.5f} | Max: {max_v:.5f} | Mean: {mean_v:.5f} | Std: {std_v:.5f}")

# 4. Compare difference between distinct content tokens
content_tokens = [t for t in tokens_info if t["token_str"] not in ("[CLS]", "[SEP]", "[PAD]", "a", "in", "")]
if len(content_tokens) >= 2:
    print("\n--- ATTENTION MAP DISSIMILARITY VERIFICATION ---")
    for i in range(len(content_tokens)):
        for j in range(i + 1, min(i + 3, len(content_tokens))):
            t1 = content_tokens[i]
            t2 = content_tokens[j]
            v1 = t1["attn_vector"]
            v2 = t2["attn_vector"]
            
            l2_dist = torch.norm(v1 - v2).item()
            cos_sim = torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            mean_abs_diff = torch.mean(torch.abs(v1 - v2)).item()
            
            print(f"Diff between \"{t1['token_str']}\" (ID {t1['token_id']}) vs \"{t2['token_str']}\" (ID {t2['token_id']}):")
            print(f"  - Euclidean Distance (L2) : {l2_dist:.5f}")
            print(f"  - Cosine Similarity       : {cos_sim:.5f}")
            print(f"  - Mean Absolute Diff      : {mean_abs_diff:.5f}")
            print(f"  - Verified Distinct?      : {'YES, DIFFERENT!' if l2_dist > 1e-4 and cos_sim < 0.999 else 'IDENTICAL'}")
