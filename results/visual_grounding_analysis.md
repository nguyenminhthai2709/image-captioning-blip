# Visual Grounding & Cross-Attention Analysis

## 1. Method Overview

Visual Grounding in the BLIP image captioning framework connects generated natural language tokens with specific spatial visual regions extracted by the Vision Transformer (ViT-B/16).

```
Image (384x384) ──> ViT-B/16 Encoder ──> 576 Visual Tokens (Keys/Values)
                                                      │
Text Tokens w_<t ──> Cross-Attention Decoder ─────────┴──> Cross-Attention Matrix [12 Layers x 12 Heads]
                                                      │
WordPiece Subwords ──> Mean Aggregation ──────────────┴──> Word Attention Vector ∈ ℝ^576
                                                      │
Z-Score Relevance ──> Spatial Grid (24x24) ───────────┴──> Bicubic Interpolation & Heatmap Overlay
```

1. **Vision Encoding**: Input images are normalized and processed by ViT-B/16 into $576$ visual patch representations ($24 \times 24$ spatial grid) plus one `[CLS]` token, yielding hidden states $\mathbf{H}_{\text{vis}} \in \mathbb{R}^{1 \times 577 \times 768}$.
2. **Text Decoder Cross-Attention**: When decoding token $w_t$, the Text Decoder computes multi-head cross-attention across all visual tokens:
   $$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   Extracting the raw cross-attention weights yields tensors of shape `[12, 12, seq_len, 577]`.
3. **Layer & Head Aggregation**: The highest 4 semantic layers (Layers 8, 9, 10, 11) are averaged across all 12 attention heads. The `[CLS]` token (index 0) is excluded, producing a sequence attention matrix of shape `[seq_len, 576]`.
4. **Z-Score Contrast Enhancement**: To eliminate background attention sinks, spatial relevance is normalized per-patch across the sequence:
   $$R(t, j) = \text{ReLU}\left(\frac{A(t, j) - \mu_j}{\sigma_j}\right)$$
5. **Word-Level Subword Aggregation**: For compound or subword tokens (e.g., `["cr", "##oche"]`), attention vectors are averaged across subword indices, linking complete human words to visual regions. Repeated words are tracked by token position index to avoid collision.
6. **Heatmap Overlay**: The $576$-dimensional vector is reshaped to $(24, 24)$, interpolated to original image dimensions via bicubic interpolation, and blended onto the RGB image.

---

## 2. Verified Implementation Details

| Parameter | Verified Value | Source |
| :--- | :--- | :--- |
| **Model** | `Salesforce/blip-image-captioning-base` / `checkpoints/frozen_vision/best_model` | Model config |
| **Input Image Resolution** | $384 \times 384$ px | `vision_config.image_size` |
| **Patch Size** | $16 \times 16$ px | `vision_config.patch_size` |
| **Number of Visual Patches** | $576$ spatial patches ($+1$ `[CLS]` token) | $(384/16)^2 = 24 \times 24$ |
| **Spatial Grid Dimensions** | $24 \times 24$ | Patch lattice |
| **Decoder Cross-Attention Source** | `decoder_outputs.cross_attentions` | Text decoder pass |
| **Decoder Layers Used** | Top 4 layers (Layers 8, 9, 10, 11) | Semantic cross-attention |
| **Head Aggregation** | Mean over 12 attention heads | Uniform mean |
| **Subword Aggregation** | Mean over constituent subword token indices | `group_subwords_into_words()` |
| **Spatial Normalization** | Sequence Z-score + $\text{ReLU}$ contrast enhancement | Denoising attention sinks |
| **Image Interpolation** | Bicubic interpolation to original $(H, W)$ | `torch.nn.functional.interpolate` |

---

## 3. Qualitative Examples

### Sample 01: 3482062809_3b694322c4.jpg

- **Scenario**: Person / Object Grounding
- **Generated Caption**: "a group of people sitting on a bench in front of a statue of a soldier"
- **Analyzed Word/Token**: "statue" (Tokens: `['statue']`, Indices: `[13]`)
- **Visualization File**: [`results/figures/visual_grounding/attention_sample_01.png`](file:///results/figures/visual_grounding/attention_sample_01.png)
- **Factual Observation**: The cross-attention visualization places relatively higher attention weight on the visual patches corresponding to the living statue performer standing on the platform when generating the token 'statue'.

---

### Sample 02: 280706862_14c30d734a.jpg

- **Scenario**: Animal / Object Attribute Grounding
- **Generated Caption**: "a black dog with a white ball in it ' s mouth"
- **Analyzed Word/Token**: "ball" (Tokens: `['ball']`, Indices: `[7]`)
- **Visualization File**: [`results/figures/visual_grounding/attention_sample_02.png`](file:///results/figures/visual_grounding/attention_sample_02.png)
- **Factual Observation**: The cross-attention visualization shows relatively concentrated attention around the dog's mouth and the white object region when decoding the token 'ball'.

---

### Sample 03: 3110649716_c17e14670e.jpg

- **Scenario**: Action / Interaction Grounding
- **Generated Caption**: "a man in a green jacket is helping another man in a black jacket"
- **Analyzed Word/Token**: "helping" (Tokens: `['helping']`, Indices: `[8]`)
- **Visualization File**: [`results/figures/visual_grounding/attention_sample_03.png`](file:///results/figures/visual_grounding/attention_sample_03.png)
- **Factual Observation**: When generating the action verb 'helping', the attention distribution is focused on the central interaction zone where the two individuals' arms and upper bodies meet.

---

### Sample 04: 3100251515_c68027cc22.jpg

- **Scenario**: Scene / Context Object Grounding
- **Generated Caption**: "a group of people holding signs in the street"
- **Analyzed Word/Token**: "signs" (Tokens: `['signs']`, Indices: `[6]`)
- **Visualization File**: [`results/figures/visual_grounding/attention_sample_04.png`](file:///results/figures/visual_grounding/attention_sample_04.png)
- **Factual Observation**: During generation of the plural noun 'signs', cross-attention is elevated across the upper-middle spatial region where protest signs and banners are held aloft.

---

### Sample 05: 1258913059_07c613f7ff.jpg

- **Scenario**: Ambiguous / Limitation Case
- **Generated Caption**: "a red white and blue building"
- **Analyzed Word/Token**: "building" (Tokens: `['building']`, Indices: `[6]`)
- **Visualization File**: [`results/figures/visual_grounding/attention_sample_05.png`](file:///results/figures/visual_grounding/attention_sample_05.png)
- **Factual Observation**: Attention is broadly distributed across the large patterned Union Jack facade in the background, showing how strong background visual textures can dominate decoder attention over foreground subjects.

---

## 4. Limitations & Interpretability Scope

- **Interpretability Signal, Not Ground-Truth Detection**: Cross-attention heatmaps illustrate which visual patch representations were queried most strongly by the language decoder during token generation. They are an interpretability tool and do not constitute ground-truth bounding box detection or object localization.
- **Attention Diffusion**: Attention distributions can be diffuse across broad background elements or contextual regions rather than sharply bounded around object silhouettes.
- **Dependence on Decoder Trajectory**: The extracted attention map is strictly conditioned on the autoregressively generated sequence and the specific token query.
