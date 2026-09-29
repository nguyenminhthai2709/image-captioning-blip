import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
model.eval()

# Select test image
img_path = Path('data/flickr8k/Images/1001773457_577c3a7d70.jpg')
img = Image.open(img_path).convert('RGB')
inputs = processor(images=img, return_tensors='pt')
pixel_values = inputs.pixel_values

print("=" * 80)
print("1. GENERATING CAPTION (Beam Search k=5)")
print("=" * 80)
with torch.no_grad():
    out_ids = model.generate(pixel_values=pixel_values, num_beams=5, max_length=32)[0]
    
generated_caption = processor.decode(out_ids, skip_special_tokens=True).strip()
print(f"Generated Caption: \"{generated_caption}\"")

print("\n" + "=" * 80)
print("2. TOKENIZATION & INPUT_IDS ALIGNMENT")
print("=" * 80)
# Tokenize the generated caption directly
caption_encoding = processor.tokenizer(generated_caption, return_tensors="pt")
input_ids = caption_encoding.input_ids[0] # [seq_len]
decoded_tokens = [processor.tokenizer.decode([t_id]).strip() for t_id in input_ids]

print(f"Total Tokens: {len(input_ids)}")
for idx, (t_id, t_str) in enumerate(zip(input_ids, decoded_tokens)):
    print(f"  [{idx:2d}] Token ID: {t_id.item():<6} | Decoded: \"{t_str}\"")

print("\n" + "=" * 80)
print("3. VISION ENCODER & TEXT DECODER CROSS-ATTENTION PASS")
print("=" * 80)
with torch.no_grad():
    # Vision Encoder
    vision_out = model.vision_model(pixel_values=pixel_values)
    image_embeds = vision_out[0] # [1, 577, 768]
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
    
    # Text Decoder with exact caption input_ids
    decoder_out = model.text_decoder(
        input_ids=input_ids.unsqueeze(0),
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        output_attentions=True,
        return_dict=True
    )

cross_attentions = decoder_out.cross_attentions
print(f"Number of Decoder Layers : {len(cross_attentions)}")
print(f"Layer Tensor Shape       : {list(cross_attentions[0].shape)}")
print("Dimension Mapping:")
print("  - Dim 0: Batch size = 1")
print("  - Dim 1: Attention Heads = 12")
print(f"  - Dim 2: Text Tokens / Query = {len(input_ids)}")
print("  - Dim 3: Visual Image Patches / Key = 577 (1 CLS + 576 Patches)")

# Stack top 4 layers (8, 9, 10, 11)
# Shape: [4, 1, 12, seq_len, 577]
top_layers = torch.stack(cross_attentions[-4:], dim=0)
# Mean across 4 layers and 12 heads: shape [seq_len, 576] (dropping CLS index 0)
attn_matrix = top_layers.mean(dim=(0, 1, 2)).squeeze(0)[:, 1:] # [seq_len, 576]

# Token-specific relevance: Z-score relative to sequence mean
seq_mean = attn_matrix.mean(dim=0, keepdim=True)
seq_std = attn_matrix.std(dim=0, keepdim=True) + 1e-6
z_relevance = torch.clamp((attn_matrix - seq_mean) / seq_std, min=0.0)

print("\n" + "=" * 80)
print("4. TOKEN-BY-TOKEN ATTENTION STATISTICS & PEAK PATCHES")
print("=" * 80)
content_tokens = []
for idx, (t_id, t_str) in enumerate(zip(input_ids, decoded_tokens)):
    raw_vec = attn_matrix[idx].numpy()
    rel_vec = z_relevance[idx].numpy()
    peak_p = int(rel_vec.argmax())
    
    if t_str not in ("[CLS]", "[SEP]", "a", "and", "on", "the", "in", "of"):
        content_tokens.append((idx, t_id.item(), t_str, raw_vec, rel_vec))
        
    print(f"Token [{idx:2d}] \"{t_str:<10}\" (ID: {t_id.item():<6}):")
    print(f"  - Raw Attn Min/Max   : {raw_vec.min():.5f} / {raw_vec.max():.5f}")
    print(f"  - Peak Patch Index   : #{peak_p:3d} (Row {peak_p // 24:2d}, Col {peak_p % 24:2d})")
    print(f"  - Relevance Mean/Max : {rel_vec.mean():.4f} / {rel_vec.max():.4f}")

print("\n" + "=" * 80)
print("5. PAIRWISE TOKEN DISSIMILARITY VERIFICATION (Mean Absolute Difference & L2)")
print("=" * 80)
for i in range(len(content_tokens)):
    for j in range(i + 1, min(i + 3, len(content_tokens))):
        i_idx, i_id, i_str, i_raw, i_rel = content_tokens[i]
        j_idx, j_id, j_str, j_raw, j_rel = content_tokens[j]
        
        mean_abs_diff = float(np.mean(np.abs(i_rel - j_rel)))
        l2_dist = float(np.linalg.norm(i_rel - j_rel))
        cos_sim = float(np.dot(i_rel, j_rel) / (np.linalg.norm(i_rel) * np.linalg.norm(j_rel) + 1e-8))
        
        print(f"Difference between \"{i_str}\" (ID {i_id}) vs \"{j_str}\" (ID {j_id}):")
        print(f"  • Mean Absolute Difference : {mean_abs_diff:.5f}")
        print(f"  • Euclidean Distance (L2)  : {l2_dist:.4f}")
        print(f"  • Cosine Similarity        : {cos_sim:.4f}")
        print(f"  • Verified Distinct?       : {'YES, PROVEN DISTINCT!' if mean_abs_diff > 0.01 else 'CLOSE'}")
