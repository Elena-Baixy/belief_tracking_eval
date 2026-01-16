# Circuit Analysis Plan — Successor Heads

## Goal
Identify **Successor Heads** in GPT-2 small that implement ordinal sequence continuation by predicting the next item in learned sequences like "Monday → Tuesday" or "1 → 2".

Note: Originally planned for pythia-14m, but used GPT-2 small due to model availability.

## Hypothesis
The successor circuit consists of specialized attention heads that:
1. **Successor Heads** attend to the current item in an ordinal sequence and retrieve the successor from a learned mapping.
2. Different heads may specialize in different ordinal types (numbers vs. days vs. months).
3. The mechanism implements a "+1" operation for items in ordinal sequences learned during pretraining.

## Method

### Phase 1: Dataset Creation ✓
- Created ordinal sequences covering days, months, numbers, letters
- 61 examples total using sequence continuation format

### Phase 2: Baseline Evaluation ✓
- Tested model's successor prediction accuracy: 80.3% overall
- Days: 83.3%, Months: 90.9%, Numbers: 94.7%, Letters: 64.0%

### Phase 3: Direct Logit Attribution ✓
- Computed each head's contribution to correct successor token's logit
- Top heads: a9.h1, a11.h8, a10.h3, a7.h11

### Phase 4: Attention Pattern Analysis ✓
- Visualized attention patterns for top heads
- a9.h1 shows 58% attention to predecessor token

### Phase 5: Ablation Experiments ✓
- Single head ablation: 1.6-3.3% drops
- Multi-head ablation: up to 14.8% drop

### Phase 6: Circuit Identification ✓
- Identified 4 key successor heads
- Saved to real_circuits_1.json

## Expected Outcomes

### If hypothesis is supported: ✓
- 4 heads show strong direct logit attribution for successor predictions
- These heads attend to the current ordinal item
- Ablating these heads degrades successor prediction
- Specialization observed (calendar heads vs. letter heads)

## Architecture Notes
- GPT-2 small: 12 layers, 12 heads per layer
- d_model = 768, d_head = 64
- Nodes format: input, a{0-11}.h{0-11}, m{0-11}
