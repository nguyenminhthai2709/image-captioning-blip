# Qualitative Analysis

## Evaluation Setup

- Dataset: Flickr8k
- Test images: 1,000
- References per image: 5
- Pretrained model: Salesforce/blip-image-captioning-base
- Fine-tuned model: checkpoints/frozen_vision/best_model/
- Decoding: Beam Search
- num_beams: 3
- max_length: 32

## Quantitative Summary

| Metric | Pretrained | Fine-Tuned | Difference |
|---|---:|---:|---:|
| BLEU-1 | 58.99 | 71.15 | +12.16 pp |
| BLEU-2 | 45.08 | 54.86 | +9.78 pp |
| BLEU-3 | 33.37 | 40.65 | +7.28 pp |
| BLEU-4 | 24.50 | 29.33 | +4.83 pp |
| METEOR | 37.19 | 44.40 | +7.21 pp |
| ROUGE-L | 49.40 | 53.25 | +3.85 pp |

## Qualitative Examples

### Sample 01: 3482062809_3b694322c4.jpg

- **Category**: Clear improvement (Scene / Entity recognition)
- **Image Path**: `data/flickr8k/Images/3482062809_3b694322c4.jpg`
- **Selection Reason**: Fine-tuned caption accurately identifies the statue, bench, and group of people, correcting the pretrained model's false detection of a skateboarder.

**Ground-Truth References (5 captions)**:
1. A group of tourists stand around as a lady puts her hand near the mouth of a statue .
2. A woman is making a statue pretend to kiss her hand beside four boys at a bench .
3. A woman posing with a statue alongside a group of boys
4. A woman with a backpack leans again a statue while a group of boys sit on a bench talking .
5. Woman gets her hand kissed by living statue street artist .

- **Pretrained BLIP**: "a man on a skateboard"
- **Fine-Tuned BLIP**: "a group of people sitting on a bench in front of a statue of a soldier"
- **Observation**: Fine-tuned caption accurately identifies the statue, bench, and group of people, correcting the pretrained model's false detection of a skateboarder.

---

### Sample 02: 280706862_14c30d734a.jpg

- **Category**: Object recognition improvement
- **Image Path**: `data/flickr8k/Images/280706862_14c30d734a.jpg`
- **Selection Reason**: Fine-tuned caption specifies the object being carried (white ball), whereas the pretrained caption only describes the dog color.

**Ground-Truth References (5 captions)**:
1. A black dog emerges from the water onto the sand , holding a white object in its mouth .
2. A black dog emerges from the water with a white ball in its mouth .
3. A black dog on a beach carrying a ball in its mouth .
4. a black dog walking out of the water with a white ball in his mouth .
5. The black dog jumps out of the water with something in its mouth .

- **Pretrained BLIP**: "the dog is black"
- **Fine-Tuned BLIP**: "a black dog with a white ball in it ' s mouth"
- **Observation**: Fine-tuned caption specifies the object being carried (white ball), whereas the pretrained caption only describes the dog color.

---

### Sample 03: 3110649716_c17e14670e.jpg

- **Category**: Action / Interaction improvement
- **Image Path**: `data/flickr8k/Images/3110649716_c17e14670e.jpg`
- **Selection Reason**: Fine-tuned caption captures the interactive action between the two individuals rather than just describing clothing.

**Ground-Truth References (5 captions)**:
1. A man helps another man tie a red ribbon onto his arm .
2. A man helps tie a red ribbon around another man 's right arm during a street parade .
3. A man is tying a red arm band around another mans arm in the street .
4. One man helps another attach a red ribbon to his forearm in the midst of a large group of people .
5. Two men stand together ; one is putting something red on his arm .

- **Pretrained BLIP**: "a man wearing a green jacket"
- **Fine-Tuned BLIP**: "a man in a green jacket is helping another man in a black jacket"
- **Observation**: Fine-tuned caption captures the interactive action between the two individuals rather than just describing clothing.

---

### Sample 04: 3100251515_c68027cc22.jpg

- **Category**: Scene / Context improvement
- **Image Path**: `data/flickr8k/Images/3100251515_c68027cc22.jpg`
- **Selection Reason**: Fine-tuned caption provides contextual detail regarding the signs and street setting, whereas pretrained caption is overly generic.

**Ground-Truth References (5 captions)**:
1. A crowd of people wearing jackets and holding signs .
2. A group of protesters picket in the road .
3. A large " green " peaceful protest is taken to the streets .
4. A protest march is passing by a white stone building .
5. Group of people walking with posters .

- **Pretrained BLIP**: "a crowd of people"
- **Fine-Tuned BLIP**: "a group of people holding signs in the street"
- **Observation**: Fine-tuned caption provides contextual detail regarding the signs and street setting, whereas pretrained caption is overly generic.

---

### Sample 05: 3072172967_630e9c69d0.jpg

- **Category**: Similar performance
- **Image Path**: `data/flickr8k/Images/3072172967_630e9c69d0.jpg`
- **Selection Reason**: Both models generate identical and accurate descriptions of the basketball game.

**Ground-Truth References (5 captions)**:
1. A player from the white and green highschool team dribbles down court defended by a player from the other team .
2. Four basketball players in action .
3. Four men playing basketball , two from each team .
4. Two boys in green and white uniforms play basketball with two boys in blue and white uniforms .
5. Young men playing basketball in a competition .

- **Pretrained BLIP**: "a group of men playing a game of basketball"
- **Fine-Tuned BLIP**: "a group of men playing a game of basketball"
- **Observation**: Both models generate identical and accurate descriptions of the basketball game.

---

### Sample 06: 1258913059_07c613f7ff.jpg

- **Category**: Failure / Limitation case
- **Image Path**: `data/flickr8k/Images/1258913059_07c613f7ff.jpg`
- **Selection Reason**: Both models focus solely on the background painted building and fail to mention the people sitting at the picnic table.

**Ground-Truth References (5 captions)**:
1. A couple of people sit outdoors at a table with an umbrella and talk .
2. Three people are sitting at an outside picnic bench with an umbrella .
3. Three people sit at an outdoor cafe .
4. Three people sit at an outdoor table in front of a building painted like the Union Jack .
5. Three people sit at a picnic table outside of a building painted like a union jack .

- **Pretrained BLIP**: "a red, white, and blue building"
- **Fine-Tuned BLIP**: "a red white and blue building"
- **Observation**: Both models focus solely on the background painted building and fail to mention the people sitting at the picnic table.

---

## Observations & Scope

- **Domain-Specific Alignment**: Fine-tuning on Flickr8k helps the model produce descriptions with more specific vocabulary, attributes, and actions commonly seen in outdoor activity datasets.
- **Similar Performance on Standard Scenarios**: In clear, standard scenes (such as common sports or animal activities), both pretrained and fine-tuned models generate identical or very similar captions.
- **Limitations**: In complex scenes with competing foreground and background visual elements, both models can focus on salient background patterns while omitting foreground human interactions.
- **Conclusion**: The qualitative analysis confirms the quantitative findings on the Flickr8k test experiment without claiming universal superiority across out-of-domain datasets.
