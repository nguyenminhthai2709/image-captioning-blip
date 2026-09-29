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

# Test image: 1000268201_693b08cb0e.jpg (little girl in pink dress)
img_path = Path('data/flickr8k/Images/1000268201_693b08cb0e.jpg')
img = Image.open(img_path).convert('RGB')
inputs = processor(images=img, return_tensors='pt')
pixel_values = inputs.pixel_values

# Enable gradients for Grad-CAM relevance
pixel_values.requires_grad_(True)

out_ids = model.generate(pixel_values=pixel_values, max_length=32)[0]
caption = processor.decode(out_ids, skip_special_tokens=True)
tokens = [processor.decode([t]).strip() for t in out_ids]
print(f"Caption: \"{caption}\"")
print(f"Tokens: {tokens}")

# Method A: Gradient-weighted relevance (Chefer et al. / Grad-CAM on Cross-Attention)
# Let's compute forward pass
vision_outputs = model.vision_model(pixel_values=pixel_values)
image_embeds = vision_outputs[0]
image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

# Hook cross-attention layer to capture attention and its gradient
dec_input_ids = out_ids.unsqueeze(0)

# Method B: Token-Specific Z-score across the sequence (removes attention sinks)
dec_out = model.text_decoder(
    input_ids=dec_input_ids,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
    output_attentions=True,
    return_dict=True
)

# cross_attentions: tuple of 12 layers [1, 12, seq_len, 577]
# Take layers 8 to 11
layers = torch.stack(dec_out.cross_attentions[-4:], dim=0) # [4, 1, 12, seq_len, 577]
# Mean across heads & selected layers: [seq_len, 576] (dropping CLS at index 0)
attn_matrix = layers.mean(dim=(0, 1, 2))[:, 1:] # [seq_len, 576]

# Token Z-score: For each patch j, calculate mean and std across all tokens in the sentence
patch_mean = attn_matrix.mean(dim=0, keepdim=True) # [1, 576]
patch_std = attn_matrix.std(dim=0, keepdim=True) + 1e-6 # [1, 576]
z_scores = (attn_matrix - patch_mean) / patch_std # [seq_len, 576]
# ReLU to keep positive specific activations
specific_relevance = torch.clamp(z_scores, min=0.0)

print("\n--- TOKEN SPECIFIC PEAK ANALYSIS WITH Z-SCORE RELEVANCE ---")
for idx, tok in enumerate(tokens):
    if tok in ("[CLS]", "[SEP]", ""):
        continue
    rel = specific_relevance[idx].detach().numpy()
    norm_rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-8)
    peak_patch = int(norm_rel.argmax())
    peak_y, peak_x = peak_patch // 24, peak_patch % 24
    print(f"Token [{idx:2d}] \"{tok:<10}\" -> Peak Patch: {peak_patch:3d} (Grid row={peak_y:2d}, col={peak_x:2d}), Max={norm_rel.max():.4f}, Mean={norm_rel.mean():.4f}")

# Check distances between specific relevance vectors
content_indices = [i for i, t in enumerate(tokens) if t not in ("[CLS]", "[SEP]", "a", "in", "")]
print("\n--- PAIRWISE DISSIMILARITY AFTER SINK REMOVAL ---")
for i in range(len(content_indices)):
    for j in range(i + 1, min(i + 3, len(content_indices))):
        idx_i, idx_j = content_indices[i], content_indices[j]
        v_i = specific_relevance[idx_i].detach()
        v_j = specific_relevance[idx_j].detach()
        l2 = torch.norm(v_i - v_j).item()
        cos_sim = torch.cosine_similarity(v_i.unsqueeze(0), v_j.unsqueeze(0)).item()
        print(f"Diff between \"{tokens[idx_i]}\" vs \"{tokens[idx_j]}\": L2={l2:.4f}, CosSim={cos_sim:.4f}")
