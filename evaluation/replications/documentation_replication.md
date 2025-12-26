# Belief Tracking Replication Documentation

## Goal

Replicate the key findings from "Language Models use Lookbacks to Track Beliefs" (Prakash et al., 2025), specifically the causal intervention experiments that localize:
1. **Answer Pointer**: The mechanism that directs the model to retrieve the correct state token
2. **Answer Payload**: The actual state value that the model outputs

## Data

### Dataset: CausalToM
- **Source**: Synthetic belief tracking scenarios in a restaurant setting
- **Structure**: Stories about two characters filling containers with different drinks
- **Templates**: Template 2 used (no explicit visibility constraints in story text)
- **Entities**:
  - 103 character names
  - 21 container types (cup, mug, bottle, etc.)
  - 23 drink types (water, coffee, wine, etc.)

### Counterfactual Generation
Two types of counterfactual pairs were generated:

1. **Pointer Counterfactuals**:
   - Reverses sentence order and uses different states
   - Tests if patching redirects the model's "pointer" to a different state

2. **Payload Counterfactuals**:
   - Uses completely different entities between clean and counterfactual
   - Tests if patching transfers the state value directly

## Method

### Model
- **Original Paper**: Llama-3-70B-Instruct (80 layers)
- **Replication**: Llama-3.1-8B-Instruct (32 layers)
- **Reason for difference**: Computational constraints (disk quota limitations prevented loading larger models)

### Interchange Intervention Analysis
For each layer L:
1. Run counterfactual prompt through model, save activation at layer L, final token position
2. Run clean prompt, patch in counterfactual activation at layer L, final token
3. Measure if output changes to target answer
4. Compute IIA (Interchange Intervention Accuracy) = fraction of samples where intervention succeeded

### Experimental Parameters
- **Seed**: 42 (for reproducibility)
- **Samples**: 20 per experiment
- **Pointer layers tested**: [0, 4, 8, 12, 16, 20, 24, 28]
- **Payload layers tested**: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

## Results

### Answer Pointer Localization
| Layer | IIA |
|-------|-----|
| 0     | 0.05 |
| 4     | 0.05 |
| 8     | 0.05 |
| 12    | 0.35 |
| 16    | **0.80** |
| 20    | 0.70 |
| 24    | 0.05 |
| 28    | 0.00 |

**Peak**: Layer 16 (50% of model depth) with IIA = 0.80

### Answer Payload Localization
| Layer | IIA |
|-------|-----|
| 0-22  | 0.00 |
| 24    | 0.30 |
| 26    | 0.70 |
| 28    | **0.90** |
| 30    | 0.90 |

**Peak**: Layers 28-30 (87-94% of model depth) with IIA = 0.90

## Analysis

### Comparison with Original Paper

| Aspect | Original (70B) | Replication (8B) |
|--------|----------------|------------------|
| Pointer peak layers | 34-52 (43-65% depth) | 16 (50% depth) |
| Payload peak layers | 56+ (70%+ depth) | 28-30 (87-94% depth) |
| Pointer before payload | Yes | Yes |
| High IIA achieved | Yes | Yes |

### Key Findings Confirmed
1. **Layer Ordering**: Pointer information is encoded before payload information in the network, consistent with the "lookback" mechanism proposed in the paper
2. **Localized Encoding**: Both pointer and payload are concentrated in specific layer ranges, not distributed uniformly
3. **High Intervention Accuracy**: Peak IIA values of 0.80-0.90 demonstrate that the interventions successfully modify model behavior

### Differences and Caveats
1. **Model Size**: 8B vs 70B parameters may affect the precision of localization
2. **Layer Count**: 32 vs 80 layers means coarser resolution in identifying exact layer ranges
3. **Sample Size**: 20 samples per experiment (original used 80 correctly-answered samples)
4. **Baseline Accuracy**: Model achieved 65% on belief tracking (some errors may affect IIA measurements)

## Conclusion

The replication successfully demonstrates the core findings of the original paper:
- Language models encode pointer and payload information in distinct, sequential layer ranges
- Interchange interventions can reliably redirect model outputs
- The lookback mechanism for belief tracking is a general pattern observable across model scales
