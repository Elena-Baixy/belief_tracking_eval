# Code Walkthrough: Successor Heads Circuit Analysis

## Overview
This document walks through the code used to identify Successor Heads in GPT-2 small.

## 1. Setup and Model Loading

```python
import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
```

Model architecture:
- 12 layers, 12 heads per layer
- d_model: 768, d_head: 64

## 2. Dataset Creation

Created ordinal sequence examples with sequence continuation format:

```python
def create_sequence_examples(seq, seq_type, context_length=3):
    examples = []
    for i in range(len(seq) - 1):
        start_idx = max(0, i - context_length + 1)
        context = seq[start_idx:i+1]
        prompt = ", ".join(context) + ","
        successor = seq[i + 1]
        examples.append({
            "prompt": prompt,
            "current": seq[i],
            "successor": successor,
            "type": seq_type
        })
    return examples
```

Example prompts:
- Days: `"Monday, Tuesday, Wednesday,"` → `"Thursday"`
- Numbers: `"1, 2, 3,"` → `"4"`

## 3. Direct Logit Attribution

Computes each head's contribution to the successor token's logit:

```python
def compute_head_logit_attribution(model, prompt, target_token):
    tokens = model.to_tokens(prompt)
    target_id = model.to_tokens(" " + target_token)[0, 1].item()
    
    _, cache = model.run_with_cache(tokens)
    
    W_U = model.W_U  # [d_model, d_vocab]
    target_direction = W_U[:, target_id]  # [d_model]
    
    head_attributions = torch.zeros(n_layers, n_heads)
    
    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        W_O = model.W_O[layer]
        z_last = z[0, -1, :, :]
        
        for head in range(n_heads):
            head_output = z_last[head] @ W_O[head]
            head_attributions[layer, head] = (head_output @ target_direction).item()
    
    return head_attributions
```

Key insight: Attribution = `head_output · W_U[:, target]`

## 4. Attention Pattern Analysis

Analyzed where each head attends:

```python
def analyze_attention_patterns(model, examples, heads):
    for ex in examples:
        _, cache = model.run_with_cache(tokens)
        
        for layer, head in heads:
            attn = cache[f"blocks.{layer}.attn.hook_pattern"]
            attn_last = attn[0, head, -1, :]  # Attention from last position
            
            # Check attention to predecessor (second-to-last token)
            attn_to_predecessor = attn_last[-2]
```

## 5. Ablation Experiments

Zero out head outputs and measure accuracy drop:

```python
def ablate_head_hook(z, hook, head_idx):
    z[:, :, head_idx, :] = 0
    return z

def test_with_ablation(model, examples, heads_to_ablate):
    hooks = []
    for layer, head in heads_to_ablate:
        hook_fn = partial(ablate_head_hook, head_idx=head)
        hooks.append((f"blocks.{layer}.attn.hook_z", hook_fn))
    
    for ex in examples:
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        # ... compute accuracy
```

## 6. Key Findings

### Top Successor Heads by Attribution
1. a9.h1: +14.95 (primary)
2. a11.h8: +13.55 (output)
3. a10.h3: +12.55 (calendar)
4. a7.h11: +10.32 (secondary)

### Attention Patterns
a9.h1 shows 58% average attention to predecessor token - consistent with successor head mechanism.

### Ablation Impact
- Single head ablation: 1.6-3.3% drop
- Top 4 heads ablated: 13.1% drop
- Top 5 heads ablated: 14.8% drop

## 7. Circuit Visualization

See `logs/head_attribution_heatmap.png` for attribution heatmap.
See `logs/attention_patterns.png` for attention pattern visualization.

## 8. Files Produced

- `real_circuits_1.json`: Final circuit definition
- `logs/plan.md`: Research plan
- `logs/documentation.md`: Full documentation
- `logs/code_walk.md`: This file
- `logs/head_attribution_heatmap.png`: Attribution visualization
- `logs/attention_patterns.png`: Attention patterns
