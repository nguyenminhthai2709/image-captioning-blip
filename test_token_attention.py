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

# Let's test on an image with distinct entities
images = list(Path('data/flickr8k/Images').glob('*.jpg'))[:5]

for img_file in images:
    img = Image.open(img_file).convert('RGB')
    inputs = processor(images=img, return_tensors='pt')
    pixel_values = inputs.pixel_values
    
    with torch.no_grad():
        out_ids = model.generate(pixel_values=pixel_values, max_length=32)[0]
        
    caption = processor.decode(out_ids, skip_special_tokens=True)
    tokens = [processor.decode([t]).strip() for t in out_ids]
    
    # Forward pass through vision encoder
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        img_embeds = vision_outputs[0] # [1, 577, 768]
        img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long)
        
        # Text decoder pass
        dec_out = model.text_decoder(
            input_ids=out_ids.unsqueeze(0),
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_atts,
            output_attentions=True,
            return_dict=True
        )
        
    # cross_attentions: 12 layers of [1, 12, seq_len, 577]
    cross_attns = dec_out.cross_attentions
    
    # We inspect cross-attention across layers 8, 9, 10, 11
    stacked = torch.stack(cross_attns[-4:], dim=0) # [4, 1, 12, seq_len, 577]
    # Mean across layers & heads: [seq_len, 577]
    avg_cross = stacked.mean(dim=(0, 1, 2)).squeeze(0)
    
    print(f"\nImage: {img_file.name}")
    print(f"Caption: \"{caption}\"")
    print(f"Tokens: {tokens}")
    
    # Filter content tokens
    content_indices = []
    for idx, (tid, tok) in enumerate(zip(out_ids, tokens)):
        if tok.lower() not in ("[cls]", "[sep]", "[pad]", "a", "an", "the", "in", "on", "of", "and", ""):
            content_indices.append((idx, tid.item(), tok))
            
    for idx, tid, tok in content_indices:
        patch_attn = avg_cross[idx, 1:].numpy() # (576,)
        # Contrast normalization: remove global mean and normalize
        norm_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min() + 1e-8)
        print(f"  Token [{idx}] (ID={tid}): \"{tok}\" -> Peak Patch: {norm_attn.argmax()}, Peak Val: {norm_attn.max():.4f}, Mean: {norm_attn.mean():.4f}, Std: {norm_attn.std():.4f}")
        
    if len(content_indices) >= 2:
        for a in range(len(content_indices)):
            for b in range(a + 1, min(a + 3, len(content_indices))):
                idx_a, tid_a, tok_a = content_indices[a]
                idx_b, tid_b, tok_b = content_indices[b]
                vec_a = avg_cross[idx_a, 1:]
                vec_b = avg_cross[idx_b, 1:]
                diff_l2 = torch.norm(vec_a - vec_b).item()
                cos_s = torch.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
                print(f"    Diff (\"{tok_a}\" vs \"{tok_b}\"): L2={diff_l2:.5f}, CosSim={cos_s:.5f}")
