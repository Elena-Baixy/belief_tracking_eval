# Successor Heads Circuit Documentation

## 1. Goal

Identify **Successor Heads** in GPT-2 small that implement ordinal sequence continuation by predicting the next item in learned sequences like "Monday → Tuesday", "1 → 2", or "A → B".

## 2. Hypothesis

The successor circuit consists of specialized attention heads that:
1. Attend to the current item in an ordinal sequence
2. Retrieve the successor from a learned mapping (X → X+1)
3. Write the successor token representation to the residual stream

We hypothesized that:
- Different heads may specialize in different ordinal types (numbers vs. days vs. months vs. letters)
- The mechanism implements a "+1" operation for items in ordinal sequences learned during pretraining

## 3. Data

### Dataset Structure
- **Total examples**: 61 sequence continuation prompts
- **Days**: 6 examples (Monday→Tuesday, Tuesday→Wednesday, etc.)
- **Months**: 11 examples (January→February, February→March, etc.)
- **Numbers**: 19 examples (1→2, 2→3, ..., 19→20)
- **Letters**: 25 examples (A→B, B→C, ..., Y→Z)

### Prompt Format
Sequence continuation format: `"Monday, Tuesday, Wednesday,"` → predict `"Thursday"`

### Baseline Accuracy
- Days: 83.3%
- Months: 90.9%
- Numbers: 94.7%
- Letters: 64.0%
- Overall: 80.3%

## 4. Method

### Phase 1: Direct Logit Attribution
For each attention head, computed the contribution to the correct successor token's logit:
```
head_contribution = (head_output @ W_U[:, target_token])
```

This measures how much each head's output directly promotes the correct successor.

### Phase 2: Attention Pattern Analysis
Analyzed where each head attends when making successor predictions:
- Measured attention to the "predecessor" token (last content token before comma)
- Measured attention to BOS token
- Expected successor heads to strongly attend to predecessor

### Phase 3: Ablation Experiments
Zeroed out candidate heads' outputs and measured accuracy drop:
- Individual head ablation
- Multi-head ablation (cumulative effects)

## 5. Results

### Top Heads by Direct Logit Attribution
| Head | Attribution | Role |
|------|-------------|------|
| a9.h1 | +14.95 | Primary successor head |
| a11.h8 | +13.55 | Output successor head |
| a10.h3 | +12.55 | Calendar successor head |
| a10.h2 | +11.24 | Supporting head |
| a7.h11 | +10.32 | Secondary successor head |

### Head Specialization by Sequence Type
- **Days/Months**: a10.h3 shows strongest attribution (+42-45)
- **Numbers**: a9.h1 dominates (+27.9)
- **Letters**: a11.h8 is primary (+19.4)

### Attention Patterns
| Head | Attn to Predecessor | Attn to BOS |
|------|---------------------|-------------|
| a9.h1 | 0.58 (avg) | 0.22 |
| a8.h8 | 0.54 (avg) | 0.24 |
| a7.h11 | 0.36 (avg) | 0.30 |
| a10.h3 | 0.19 (avg) | 0.53 |

a9.h1 and a8.h8 show strongest attention to predecessor token.

### Ablation Results
| Ablated Heads | Overall Drop | Notes |
|---------------|--------------|-------|
| a9.h1 | 3.3% | Letters most affected |
| a7.h11 | 3.3% | Letters most affected |
| a10.h3 | 1.6% | Days drop 16.7% |
| Top 4 heads | 13.1% | Compounding effect |
| Top 5 heads | 14.8% | Compounding effect |

## 6. Analysis

### Support for Hypothesis
The hypothesis is **partially supported**:

✅ **Confirmed**:
- Clear successor heads exist (a9.h1, a10.h3, a7.h11, a11.h8)
- Head specialization by sequence type observed
- Ablation causes accuracy drops

⚠️ **Nuanced findings**:
- Individual head ablation has modest effects (distributed computation)
- Multi-head ablation shows compounding effects (circuit redundancy)
- Numbers are most robust to ablation (may use simpler tokenization patterns)

### Circuit Interpretation
The identified circuit consists of 4 heads that work together:
1. **a9.h1 (L9, H1)**: Primary successor head with strong attention to predecessor
2. **a10.h3 (L10, H3)**: Specializes in calendar sequences (days/months)
3. **a7.h11 (L7, H11)**: General-purpose successor processing
4. **a11.h8 (L11, H8)**: Late-layer output head for letter sequences

## 7. Next Steps

1. **Larger models**: Test if same heads appear in GPT-2 medium/large
2. **Other ordinal types**: Roman numerals, ordinal words (first, second, third)
3. **Causal intervention**: Activation patching to verify causal role
4. **OV circuit analysis**: Examine output-value matrices for successor mappings
5. **Composition analysis**: How do earlier layers feed into successor heads?

## 8. Main Takeaways

1. **Successor Heads exist**: GPT-2 has dedicated attention heads for ordinal continuation
2. **Specialization**: Different heads handle different sequence types (calendar vs. letters)
3. **Distributed computation**: The circuit is somewhat distributed with redundancy
4. **Position in network**: Successor heads appear in layers 7-11 (later layers)
5. **Attention pattern**: True successor heads attend strongly to the predecessor token

### Implications for Interpretability
- Ordinal reasoning is implemented via learnable head-level computations
- The circuit is interpretable: attention to X → output X+1
- This is analogous to but distinct from induction heads (which copy patterns)
