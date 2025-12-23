# Plan
## Objective
Analyze how language models internally represent and track beliefs of characters, particularly when those beliefs may differ from reality, using causal mediation and abstraction methods to uncover systematic computational mechanisms.

## Hypothesis
1. Language models use a lookback mechanism to track beliefs, where reference information is copied to two locations (address and pointer) enabling later retrieval of important information when needed.
2. The model assigns ordering IDs to character, object, and state tokens, then binds character-object-state triples together by co-locating their reference information in low-rank subspaces.
3. A binding lookback retrieves the correct state OI using character and object OIs, and an answer lookback retrieves the corresponding state token.
4. When visibility information is provided, a visibility lookback uses a visibility ID to retrieve information about the observed character and update the observing character's beliefs.

## Methodology
1. Construct CausalToM dataset with simple stories involving two characters interacting with objects, creating counterfactual pairs for causal analysis. Analyze Llama-3-70B-Instruct and Llama-3.1-405B-Instruct on 80 correctly-answered samples.
2. Use causal mediation analysis with interchange interventions to trace information flow from key input tokens to final output across layers.
3. Apply causal abstraction to hypothesize a high-level causal model of belief tracking, then align its variables with internal activations using targeted interchange interventions.
4. Use Desiderata-based Component Masking to identify low-rank subspaces encoding specific causal variables, measuring alignment via interchange intervention accuracy (IIA).

## Experiments
### Localizing Answer Payload
- What varied: Layers intervened on at final token position with counterfactual swapping character/object/state order and state values
- Metric: Interchange intervention accuracy (IIA) measuring whether output changes from original answer to counterfactual answer
- Main result: Answer payload (state token value) localizes to final token residual stream after layer 56 with near-perfect IIA

### Localizing Answer Pointer
- What varied: Layers intervened on at final token with counterfactual pointer referencing different state than original
- Metric: IIA measuring whether output changes to new state (neither original nor counterfactual answer)
- Main result: Answer pointer information encoded at final token layers 34-52, redirecting to different state when patched

### Localizing Binding Address and Payload
- What varied: Swap residual vectors at state tokens between original and counterfactual with sentence order swapped
- Metric: IIA measuring whether output flips to the other state token in original input
- Main result: Strongest alignment occurs between layers 33-38 at state token residual stream containing both address and payload

### Localizing Binding Source Reference
- What varied: Interchange character and object token residual streams while freezing state token residual stream
- Metric: IIA measuring whether output changes to alternate state token
- Main result: Source reference (character and object OIs) encoded in character and object tokens layers 20-34

### Localizing Visibility Source Reference
- What varied: Intervene on visibility sentence tokens swapping visibility from unobserved to observed
- Metric: IIA measuring whether output changes from unknown to observed answer
- Main result: Visibility ID source encoded in visibility sentence layers 10-23, then splits into address and pointer copies

### Localizing Visibility Payload and Address+Pointer
- What varied: Intervene on (1) lookback tokens only, (2) both visibility sentence and lookback tokens with visibility flipped counterfactual
- Metric: IIA measuring output change from unknown to observed character's answer
- Main result: Payload aligns after layer 31 at lookback tokens; combined address+pointer intervention shows alignment layers 24-31 enabling QK-circuit formation