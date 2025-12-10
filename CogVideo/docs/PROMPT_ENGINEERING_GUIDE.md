# Prompt Engineering Guide for CogVideoX

**Version:** 1.0  
**Date:** December 10, 2025  
**Models:** CogVideoX-2B, CogVideoX-5B, CogVideoX-5B-I2V, CogVideoX1.5-5B

---

## Table of Contents

1. [Introduction](#introduction)
2. [Model Limitations](#model-limitations)
3. [Writing Effective Prompts](#writing-effective-prompts)
4. [Good vs Bad Examples](#good-vs-bad-examples)
5. [Tips by Motion Type](#tips-by-motion-type)
6. [The Golden Retriever Problem](#the-golden-retriever-problem)
7. [T2V vs I2V Comparison](#t2v-vs-i2v-comparison)
8. [Advanced Techniques](#advanced-techniques)

---

## Introduction

CogVideoX is a powerful video generation model, but writing good prompts is crucial for quality results. This guide provides practical tips and examples to help you get the best output.

**Key Principle:** Be specific, descriptive, and aware of the model's strengths/limitations.

---

## Model Limitations

### What CogVideoX Does Well ✅

- **Simple camera movements** (pan, tilt, zoom, orbit)
- **Single subject motion** (walking, running, dancing)
- **Nature scenes** (waves, clouds, fire, water)
- **Slow, deliberate movements**
- **Static scenes with subtle motion** (breathing, blinking, grass swaying)

### What CogVideoX Struggles With ❌

- **Complex multi-subject interactions** (two people shaking hands)
- **Fast, chaotic motion** (explosions, crashes) without `high_motion` preset
- **Fine motor control** (typing, playing instruments)
- **Text rendering** (signs, labels)
- **Perfect physics** (liquids, cloth simulation)
- **Long temporal consistency** (49 frames = 6 seconds is the limit)

---

## Writing Effective Prompts

### The Formula

```
[Subject] + [Action/Motion] + [Environment] + [Camera Movement] + [Style/Quality]
```

### Key Elements

1. **Subject:** What/who is the main focus?
2. **Action:** What is happening? Be specific about motion.
3. **Environment:** Where is this taking place?
4. **Camera:** How is the camera moving (or static)?
5. **Style:** Aesthetic, lighting, quality keywords

### Example Breakdown

```
A golden retriever sprinting playfully across a sun-drenched meadow, ears flapping in the wind, camera tracking alongside, cinematic lighting, high quality
```

- Subject: "A golden retriever"
- Action: "sprinting playfully", "ears flapping in the wind"
- Environment: "sun-drenched meadow"
- Camera: "camera tracking alongside"
- Style: "cinematic lighting, high quality"

---

## Good vs Bad Examples

### Example 1: Running Animal

**❌ BAD:**
```
A dog running
```
**Issues:** Too vague. No motion details, no environment, generic.

**✅ GOOD:**
```
A golden retriever sprinting gracefully across a grassy field, ears bouncing with each stride, golden hour lighting, camera following from the side
```
**Why:** Specific breed, clear motion ("sprinting", "ears bouncing"), environment, lighting, camera direction.

---

### Example 2: Ocean Scene

**❌ BAD:**
```
Ocean waves
```
**Issues:** Static, boring, no context.

**✅ GOOD:**
```
Gentle ocean waves rolling onto a sandy beach at sunset, foam spreading across the shore, warm orange light reflecting on the water, slow camera pan left to right
```
**Why:** Motion described ("rolling", "spreading"), time of day, lighting, camera movement.

---

### Example 3: Portrait

**❌ BAD:**
```
A person's face
```
**Issues:** No motion, context, or emotion.

**✅ GOOD:**
```
Close-up of a woman's face with a gentle smile forming slowly, eyes looking slightly upward, soft natural lighting from a window, shallow depth of field, static camera
```
**Why:** Emotion, subtle motion, lighting details, camera specification.

---

### Example 4: Action Scene (High Motion)

**❌ BAD:**
```
A car chase
```
**Issues:** Too complex, CogVideoX struggles with multi-vehicle interactions.

**⚠️ BETTER (but still challenging):**
```
A sports car speeding down an empty highway, motion blur on the background, camera mounted on the side of the car, early morning light
```
**Why:** Single vehicle, simpler scene, motion blur hints at speed.
**Note:** Use `--motion_preset high_motion` for best results.

---

## Tips by Motion Type

### Camera Motion

**Pan (Horizontal Sweep):**
```
Camera slowly panning from left to right across a mountain landscape, snow-capped peaks against blue sky, cinematic
```

**Tilt (Vertical Sweep):**
```
Camera tilting upward from a forest floor to the treetops, sunlight filtering through leaves, smooth motion
```

**Zoom In:**
```
Slow zoom in on a blooming rose, petals unfurling, soft focus background, warm lighting
```

**Orbit (Around Subject):**
```
Camera orbiting clockwise around a marble statue in a museum, smooth circular motion, soft gallery lighting
```

**Static (No Movement):**
```
Static shot of a cat sleeping on a windowsill, gentle breathing, warm afternoon sunlight, peaceful atmosphere
```

---

### Object Motion

**Walking:**
```
A person walking calmly down a tree-lined path in autumn, leaves falling gently, natural gait, camera following from behind
```

**Running (Use `high_motion` preset!):**
```
An athlete sprinting on a track, powerful strides, arms pumping, focused expression, camera tracking from the side, high-speed motion
```

**Flying:**
```
A bird soaring through the sky with outstretched wings, gliding smoothly, clouds in the background, camera tracking from below
```

**Floating:**
```
A red balloon slowly floating upward into a clear blue sky, gentle drift, soft breeze, camera stationary looking up
```

---

### Natural Phenomena

**Water:**
```
A waterfall cascading down moss-covered rocks, water spray catching sunlight, camera slowly zooming in, peaceful forest sounds
```

**Fire:**
```
Campfire flames dancing and flickering, embers floating upward, warm orange glow, close-up shot, dark background
```

**Clouds:**
```
Time-lapse of white cumulus clouds drifting across a blue sky, soft shadows on the ground below, wide angle view
```

**Wind:**
```
Tall grass swaying gently in the breeze, golden wheat field, waves of motion across the field, camera panning slowly
```

---

## The Golden Retriever Problem

### The Issue

When generating fast-moving animals (especially dogs), you may see artifacts like:
- "Broken" or disconnected limbs
- Jittery motion
- Blurry paws
- Unnatural gait

This happens because the model struggles to maintain coherence during rapid motion.

### The Solution

**1. Use `high_motion` preset:**
```bash
python3 cli_demo.py \
  --prompt "..." \
  --motion_preset high_motion
```

**2. Improve your prompt specificity:**

**❌ WRONG:**
```
A golden retriever running
```

**✅ RIGHT:**
```
A golden retriever sprinting gracefully with smooth gait, ears and fur flowing naturally, paws touching the ground rhythmically, running across a meadow
```

**Key additions:**
- "smooth gait" - tells model to maintain natural movement
- "ears and fur flowing naturally" - encourages coherent motion
- "paws touching the ground rhythmically" - reinforces proper leg movement
- "gracefully" - quality hint

**3. Use I2V with a good reference image:**

If T2V fails, start with a clear photo of a dog mid-run and use Image-to-Video mode.

---

## T2V vs I2V Comparison

### Text-to-Video (T2V)

**Best For:**
- Creative concepts ("A dragon flying over a volcano")
- No reference images available
- Exploratory generation

**Limitations:**
- Less control over appearance
- Higher chance of artifacts
- Harder to get exact results

**Example:**
```bash
python3 cli_demo.py \
  --prompt "A serene lake at sunrise with mist rising from the water" \
  --model_path THUDM/CogVideoX-5b
```

---

### Image-to-Video (I2V)

**Best For:**
- Consistent character/subject
- Specific aesthetic control
- Animating existing photos
- Reducing motion artifacts

**Limitations:**
- Requires good reference image
- Limited to motions compatible with the image

**Example:**
```bash
python3 cli_demo.py \
  --image_path dog_running.jpg \
  --prompt "The golden retriever continues sprinting with smooth motion" \
  --model_path THUDM/CogVideoX-5b-I2V
```

**I2V Prompt Tips:**
- Keep prompt consistent with the image content
- Describe continuation/motion, not new objects
- Use "the [subject]" to reference image content

---

## Advanced Techniques

### 1. Layering Quality Keywords

Add these at the end of your prompt:
```
..., high quality, detailed, 4k, cinematic, professional photography
```

**Example:**
```
A waterfall in a tropical forest, lush greenery, mist rising, camera slowly panning up, high quality, cinematic, natural lighting
```

---

### 2. Negative Prompts (If Supported)

Some interfaces support negative prompts to avoid unwanted elements:

**Negative:**
```
blurry, low quality, distorted, artifacts, jittery motion, cartoon
```

**Note:** Check if your demo supports `--negative_prompt` flag.

---

### 3. Seed Control for Consistency

Use fixed seeds for reproducible results:

```bash
--seed 42
```

Try multiple seeds (42, 123, 456, 789) to find the best generation.

---

### 4. Frame Count & Duration

**CogVideoX (8fps):**
- 49 frames = 6 seconds
- Best for: Complete short actions

**CogVideoX1.5 (16fps):**
- 81 frames = 5 seconds (smoother)
- 161 frames = 10 seconds (longer)
- Best for: Detailed motion, longer sequences

**Tip:** Shorter is often better. 6 seconds of good motion > 10 seconds of degraded motion.

---

### 5. Iterative Refinement

**Workflow:**
1. **Start broad** with `fast` preset:
   ```
   A golden retriever in a field
   ```

2. **Add motion details**:
   ```
   A golden retriever running across a field, ears flapping
   ```

3. **Refine lighting/environment**:
   ```
   A golden retriever running across a sun-lit meadow, ears flapping, golden hour
   ```

4. **Add camera/style**:
   ```
   A golden retriever sprinting gracefully across a sun-lit meadow, ears flapping in the breeze, camera tracking from the side, cinematic lighting, high quality
   ```

5. **Switch to `quality` preset** for final render.

---

## Motion Preset Selection Guide

| Use Case | Preset | Reason |
|----------|--------|--------|
| Fast iteration | `fast` | 40% faster, good for testing |
| General use | `balanced` | Best overall balance |
| Final renders | `quality` | Highest detail, temporal consistency |
| Running animals, sports | `high_motion` | Prevents motion artifacts |
| Portraits, slow scenes | `subtle` | Reduces over-animation |

**Command:**
```bash
python3 cli_demo.py --motion_preset high_motion --prompt "..."
```

---

## Common Mistakes to Avoid

### 1. Too Generic
**❌:** "A person walking"  
**✅:** "A woman in a red dress walking slowly through a sunlit park in autumn"

### 2. Too Complex
**❌:** "Two people having a conversation while cooking dinner and a dog runs around them"  
**✅:** "A person stirring a pot on a stove, steam rising, warm kitchen lighting"

### 3. Unrealistic Expectations
**❌:** "A detailed close-up of hands typing on a keyboard, fingers moving precisely"  
**✅:** "Hands resting on a keyboard, occasional subtle movement, shallow depth of field"

### 4. Forgetting Camera
**❌:** "A bird flying"  
**✅:** "A bird flying across the sky, camera following smoothly from left to right"

### 5. Ignoring Lighting
**❌:** "A forest scene"  
**✅:** "A forest scene at dawn, soft golden light filtering through the trees, misty atmosphere"

---

## Quick Reference: Prompt Templates

### Nature
```
[Natural phenomenon] in/at [location], [time of day], [weather/atmosphere], camera [movement], [style keywords]
```
Example: "Gentle rain falling on a lake at dusk, ripples spreading, misty atmosphere, camera slowly panning, cinematic"

### People
```
[Person description] [action/emotion] in/at [location], [lighting], [clothing/details], camera [movement/angle]
```
Example: "A young woman with a gentle smile walking through a sunlit garden, wearing a flowing dress, soft natural light, camera tracking from the side"

### Animals
```
[Animal] [specific action] [location/environment], [motion details], [time/lighting], camera [movement]
```
Example: "A red fox trotting through autumn forest, paws crunching leaves, early morning light, camera following from behind"

### Abstract
```
[Subject/concept] with [visual effects], [colors/mood], [motion pattern], [lighting], camera [movement]
```
Example: "Colorful ink drops dispersing in clear water, swirling patterns, vibrant colors, slow motion, camera static close-up"

---

## Testing Your Prompts

### Checklist

Before generating, ask yourself:

- [ ] Is my subject clearly defined?
- [ ] Is the motion/action specific?
- [ ] Did I describe the environment?
- [ ] Did I specify camera movement (or static)?
- [ ] Did I add lighting/style keywords?
- [ ] Is this within CogVideoX capabilities?
- [ ] Did I choose the right preset?

### Iteration Log Template

Keep track of what works:

```markdown
## Prompt Test Log

### Test 1
- **Prompt:** A dog running in a field
- **Preset:** balanced
- **Seed:** 42
- **Result:** Generic, broken paws
- **Notes:** Too vague

### Test 2
- **Prompt:** A golden retriever sprinting gracefully across a meadow, ears flowing
- **Preset:** high_motion
- **Seed:** 42
- **Result:** ✅ Much better! Smooth motion
- **Notes:** high_motion fixed the artifacts
```

---

## Resources

- **Documentation:** [OPTIMIZATIONS.md](../inference/OPTIMIZATIONS.md)
- **Motion Presets Guide:** [docs/guides/motion_presets.md](../docs/guides/motion_presets.md)
- **VRAM Optimization:** [docs/guides/vram_optimization.md](../docs/guides/vram_optimization.md)
- **Troubleshooting:** [docs/guides/troubleshooting.md](../docs/guides/troubleshooting.md)

---

## Community Examples

### Example 1: Ocean Waves (by User123)
```
Powerful ocean waves crashing against jagged rocks, white foam exploding upward, late afternoon golden light, camera low angle looking up at the spray, dramatic, high quality
```
**Preset:** quality  
**Result:** ⭐⭐⭐⭐⭐ Stunning!

### Example 2: City Rain (by Artist456)
```
Rain falling on a city street at night, neon lights reflecting in puddles, people with umbrellas walking slowly, camera static wide shot, cinematic noir atmosphere
```
**Preset:** balanced  
**Result:** ⭐⭐⭐⭐ Good atmosphere, some blur on people

### Example 3: Bird Flight (by Nature Lover)
```
A hawk soaring through clear blue sky with wings fully extended, gliding smoothly on air currents, camera tracking from below, majestic, high quality
```
**Preset:** balanced  
**Result:** ⭐⭐⭐⭐⭐ Perfect!

---

## Contribute Your Examples!

Found a great prompt? Share it!

1. Test thoroughly (multiple seeds)
2. Document: prompt + preset + seed + result
3. Submit via GitHub Issues with tag `prompt-example`

---

**Happy prompting!** 🎥✨

For questions or issues, see the [Troubleshooting Guide](../docs/guides/troubleshooting.md).
