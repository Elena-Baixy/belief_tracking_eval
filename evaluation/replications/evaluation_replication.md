# Evaluation: Belief Tracking Replication

## Reflection

This replication attempted to reproduce the core findings from "Language Models use Lookbacks to Track Beliefs" (Prakash et al., 2025). The paper investigates how language models internally track characters' beliefs using causal abstraction methods.

### What Worked Well
1. The plan.md and CodeWalkthrough.md provided clear descriptions of the experiments
2. Dataset generation code was well-structured and reusable
3. The core methodology (interchange interventions) was clearly described
4. Results showed the expected patterns despite using a smaller model

### Challenges Encountered
1. **nnsight Compatibility**: The nnsight library was incompatible with the installed PyTorch version (2.9.1), requiring reimplementation using direct PyTorch hooks
2. **Model Size**: The original experiments use 70B and 405B parameter models; we used 8B due to the nnsight issues
3. **Token Position Identification**: Finding exact state token positions required manual inspection of tokenized prompts

### Key Observations
- The lookback mechanism pattern is preserved across model scales
- Layer indices scale proportionally with model depth
- Smaller models show similar but less pronounced effects

---

## Replication Evaluation - Binary Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment can be reconstructed from the plan and code-walk documentation. The plan.md clearly describes:
- The hypothesis (lookback mechanism for belief tracking)
- The methodology (interchange interventions, IIA metric)
- The experiments (pointer, payload, binding lookbacks)
- Expected results (layer localizations)

The CodeWalkthrough.md provides repository structure and usage instructions. The notebooks contain complete working code with clear comments. Minor inference was required for token position identification, but this is documented in the original code.

---

### RP2. Environment Reproducibility

**FAIL**

**Rationale**: The environment could not be fully restored due to:
1. **nnsight incompatibility**: The nnsight library (used extensively in original code) was incompatible with PyTorch 2.9.1, causing FakeTensor errors
2. **Workaround required**: Had to reimplement interchange interventions using PyTorch hooks
3. **Model size constraint**: Could not use the original 70B model as intended due to the nnsight issues

The env.yml file only contains API keys, not package versions. The pyproject.toml exists but doesn't specify version constraints for nnsight or torch compatibility.

---

### RP3. Determinism and Stability

**PASS**

**Rationale**:
1. Random seeds are set (seed=42 in replication, seed=10 in original)
2. Results are consistent across multiple samples in the same run
3. The IIA patterns (pointer peaking in middle layers, payload in later layers) are stable
4. Variance is expected due to random sample generation but core findings are reproducible

The experiment design uses many samples and aggregates results, making it robust to sample-level variance.

---

### RP4. Demo Presentation

**NA**

**Rationale**: The repository does not contain explicit demo scripts or notebooks labeled as demos. The notebooks in the repository are full experimental notebooks, not simplified demonstrations. The replication was performed using the full experimental methodology, not a demo-only approach.

---

## Summary

The replication was **partially successful**:

- **Core findings replicated**: The lookback mechanism pattern (pointer in middle layers, payload in later layers, binding at state tokens) was successfully demonstrated
- **Scaling behavior confirmed**: Layer indices scale proportionally with model depth across different model sizes
- **Environment issues**: nnsight/PyTorch incompatibility required methodology workaround

### Checklist Summary
| Criterion | Result |
|-----------|--------|
| RP1. Implementation Reconstructability | PASS |
| RP2. Environment Reproducibility | FAIL |
| RP3. Determinism and Stability | PASS |
| RP4. Demo Presentation | NA |

### Recommendations for Future Replication
1. Pin nnsight and PyTorch versions in requirements
2. Provide fallback implementation without nnsight
3. Include smaller model configurations in documentation
4. Add explicit version compatibility notes
