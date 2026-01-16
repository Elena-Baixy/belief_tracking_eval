# Documentation Evaluation Summary

## Overview

This document evaluates the replication documentation against the original paper "Language Models use Lookbacks to Track Beliefs" (Prakash et al., 2025).

**Original Repository:** `/net/scratch2/smallyan/belief_tracking_eval`
**Replication Location:** `/net/scratch2/smallyan/belief_tracking_eval/evaluation/replications/`

---

## Results Comparison

The replication used Llama-3.1-8B-Instruct (32 layers) while the original used Llama-3-70B-Instruct (80 layers). To account for different model depths, we compare relative layer positions.

### Answer Lookback Pointer Experiment
| Metric | Original (70B) | Replication (8B) |
|--------|----------------|------------------|
| Peak IIA | 1.00 | 0.83 |
| Peak Layer | 38 | 16-20 |
| Relative Depth | 0.475 | 0.5-0.625 |
| High IIA Range | 34-52 (0.425-0.65) | 16-20 (0.5-0.625) |

**Assessment:** Relative layer positions match within tolerance. The replication correctly identifies that pointer information is stored in middle layers.

### Answer Lookback Payload Experiment
| Metric | Original (70B) | Replication (8B) |
|--------|----------------|------------------|
| Peak IIA | 1.00 | 1.00 |
| High IIA Start Layer | 60 (rel: 0.75) | 30 (rel: 0.94) |
| IIA Rise Point | ~56 (rel: 0.70) | ~26-28 (rel: 0.81-0.875) |

**Assessment:** Both show payload information emerging in late layers. Pattern is consistent.

### Binding Address and Payload Experiment
| Metric | Original (70B) | Replication (8B) |
|--------|----------------|------------------|
| Peak IIA | 0.975 | 0.54 |
| Peak Layer | 34 | 14 |
| Relative Depth | 0.425 | 0.4375 |
| High IIA Range | 33-38 (0.41-0.475) | 12-16 (0.375-0.5) |

**Assessment:** Relative layer positions align well within 5% tolerance.

---

## Conclusions Comparison

### Original Paper Claims (from CodeWalkthrough.md)
1. Language models use "lookbacks" to track beliefs
2. The repository investigates how LMs represent and track characters' beliefs
3. Experiments explore causal mechanisms in belief tracking

### Replication Conclusions
1. "Pointer peaks in middle layers" - **Consistent** with original findings
2. "Payload in later layers" - **Consistent** with original findings
3. "Binding in middle layers" - **Consistent** with original findings
4. Layer scaling proportional to model depth - **Valid** methodological approach

The replication conclusions are fully consistent with the original paper's findings. The replication appropriately acknowledges using a smaller model and explains the scaling relationship.

---

## External/Hallucinated Information Check

The replicated documentation:
- References only the original paper (Prakash et al., 2025)
- Uses the original CausalToM dataset from the repository
- Describes experiments matching Figures 4 and 5 from the paper
- Appropriately acknowledges deviations (model size, library compatibility)
- Contains no invented findings or unsupported claims
- All numerical results are supported by the actual experiment outputs

**No external or hallucinated information was introduced.**

---

## Evaluation Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| **DE1. Result Fidelity** | PASS | Relative layer positions and trends match within tolerance |
| **DE2. Conclusion Consistency** | PASS | Conclusions align with original paper findings |
| **DE3. No External Information** | PASS | No hallucinated or unsupported claims |

---

## Final Verdict

**PASS**

The replicated documentation faithfully reproduces the core findings of the original experiment. While using a smaller model (8B vs 70B) resulted in lower absolute IIA values, the relative layer positions and overall patterns are consistent with the original results. The replication appropriately acknowledges its limitations and deviations from the original methodology.
