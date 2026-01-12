"""
Replication Script: Language Models use Lookbacks to Track Beliefs

This script replicates the key experiments from the paper.
"""

import json
import os
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the data paths
DATA_DIR = "/net/scratch2/smallyan/belief_tracking_eval/data"
OUTPUT_DIR = "/net/scratch2/smallyan/belief_tracking_eval/evaluation/replications"

# Load synthetic entities
with open(os.path.join(DATA_DIR, "synthetic_entities", "characters.json"), "r") as f:
    ALL_CHARACTERS = json.load(f)

with open(os.path.join(DATA_DIR, "synthetic_entities", "bottles.json"), "r") as f:
    ALL_OBJECTS = json.load(f)

with open(os.path.join(DATA_DIR, "synthetic_entities", "drinks.json"), "r") as f:
    ALL_STATES = json.load(f)

# Load story templates
with open(os.path.join(DATA_DIR, "story_templates.json"), "r") as f:
    STORY_TEMPLATES = json.load(f)

print(f"Number of characters: {len(ALL_CHARACTERS)}")
print(f"Number of objects: {len(ALL_OBJECTS)}")
print(f"Number of states: {len(ALL_STATES)}")


@dataclass
class ReplicatedSample:
    """A sample from the CausalToM dataset."""
    template_idx: int
    characters: list
    objects: list
    states: list
    story: str = None
    character_belief: list = None

    def __post_init__(self):
        if len(self.characters) == 1:
            self.characters.append("<N/A>")

        assert len(set(self.states)) == len(self.states), "States must be unique"
        assert len(set(self.objects)) == len(self.objects), "Objects must be unique"
        assert len(set(self.characters)) == len(self.characters), "Characters must be unique"

        self._build_story()

    def _build_story(self):
        template = STORY_TEMPLATES["templates"][self.template_idx]
        self.story = template["context"]

        self.world_state = {
            self.objects[0]: self.states[0],
            self.objects[1]: self.states[1]
        }

        self.character_belief = [
            self.world_state.copy(),
            self.world_state.copy()
        ]

        if self.template_idx in [0, 2, 3]:
            self.character_belief[0][self.objects[1]] = "unknown"
            self.character_belief[1][self.objects[0]] = "unknown"
        elif self.template_idx == 1:
            self.character_belief[1][self.objects[0]] = "unknown"

        self._substitute_entities()

    def _substitute_entities(self):
        placeholders = STORY_TEMPLATES["placeholders"]["entity"]

        for i, char in enumerate(self.characters):
            self.story = self.story.replace(placeholders["character"][i], char)

        for i, obj in enumerate(self.objects):
            self.story = self.story.replace(placeholders["container"][i], obj)

        for i, state in enumerate(self.states):
            self.story = self.story.replace(placeholders["state"][i], state)

        assert "<" not in self.story and ">" not in self.story, "Unreplaced placeholders found"


class ReplicatedDataset:
    """Dataset for CausalToM experiments."""

    INSTRUCTION = (
        "1. Track the belief of each character as described in the story. "
        "2. A character's belief is formed only when they perform an action themselves "
        "or can observe the action taking place. "
        "3. A character does not have any beliefs about the container and its contents "
        "which they cannot observe. "
        "4. To answer the question, predict only what is inside the queried container, "
        "strictly based on the belief of the character, mentioned in the question. "
        "5. If the queried character has no belief about the container in question, "
        "then predict 'unknown'. "
        "6. Do not predict container or character as the final output."
    )

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int, set_character: int = None, set_container: int = None):
        sample = self.samples[idx]

        char_idx = random.choice([0, 1]) if set_character is None else set_character
        obj_idx = random.choice([0, 1]) if set_container is None else set_container

        q_character = sample.characters[char_idx]
        q_object = sample.objects[obj_idx]

        belief_states = sample.character_belief[char_idx]
        answer = belief_states.get(q_object, "unknown")

        question_template = STORY_TEMPLATES["templates"][sample.template_idx]["question"]
        question = question_template.replace(
            STORY_TEMPLATES["placeholders"]["question"]["character"], q_character
        ).replace(
            STORY_TEMPLATES["placeholders"]["question"]["container"], q_object
        )

        prompt = f"Instruction: {self.INSTRUCTION.strip()}\n\n"
        prompt += f"Story: {sample.story.strip()}\n"
        prompt += f"Question: {question}\n"
        prompt += "Answer:"

        return {
            "characters": sample.characters,
            "objects": sample.objects,
            "states": sample.states,
            "story": sample.story,
            "question": question,
            "target": answer,
            "prompt": prompt,
            "character_idx": char_idx,
            "object_idx": obj_idx,
            "template_idx": sample.template_idx
        }


def create_evaluation_samples(n_samples: int = 20):
    """Create samples for model evaluation."""
    samples = []
    for _ in range(n_samples):
        characters = random.sample(ALL_CHARACTERS, 2)
        objects = random.sample(ALL_OBJECTS, 2)
        states = random.sample(ALL_STATES, 2)

        sample = ReplicatedSample(
            template_idx=2,
            characters=characters,
            objects=objects,
            states=states
        )
        samples.append(sample)

    return ReplicatedDataset(samples)


def generate_pointer_counterfactuals(n_samples: int = 20):
    """Generate counterfactual samples for pointer localization."""
    samples = []

    for _ in range(n_samples):
        characters = random.sample(ALL_CHARACTERS, 2)
        objects = random.sample(ALL_OBJECTS, 2)
        states = random.sample(ALL_STATES, 2)

        clean_sample = ReplicatedSample(
            template_idx=2,
            characters=characters,
            objects=objects,
            states=states
        )

        new_states = random.sample(ALL_STATES, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(ALL_STATES, 2)

        cf_sample = ReplicatedSample(
            template_idx=2,
            characters=list(reversed(characters)),
            objects=list(reversed(objects)),
            states=new_states
        )

        clean_dataset = ReplicatedDataset([clean_sample])
        cf_dataset = ReplicatedDataset([cf_sample])

        random_choice = random.choice([0, 1])

        clean_item = clean_dataset.__getitem__(
            0, set_character=random_choice, set_container=random_choice
        )
        cf_item = cf_dataset.__getitem__(
            0, set_character=1 ^ random_choice, set_container=1 ^ random_choice
        )

        target = " " + states[1 ^ random_choice]

        samples.append({
            "clean_prompt": clean_item["prompt"],
            "clean_ans": clean_item["target"],
            "counterfactual_prompt": cf_item["prompt"],
            "counterfactual_ans": cf_item["target"],
            "target": target
        })

    return samples


def generate_payload_counterfactuals(n_samples: int = 20):
    """Generate counterfactual samples for payload localization."""
    samples = []

    for _ in range(n_samples):
        characters1 = random.sample(ALL_CHARACTERS, 2)
        objects1 = random.sample(ALL_OBJECTS, 2)
        states1 = random.sample(ALL_STATES, 2)

        clean_sample = ReplicatedSample(
            template_idx=2,
            characters=characters1,
            objects=objects1,
            states=states1
        )

        characters2 = random.sample(ALL_CHARACTERS, 2)
        objects2 = random.sample(ALL_OBJECTS, 2)
        states2 = random.sample(ALL_STATES, 2)

        cf_sample = ReplicatedSample(
            template_idx=2,
            characters=characters2,
            objects=objects2,
            states=states2
        )

        clean_dataset = ReplicatedDataset([clean_sample])
        cf_dataset = ReplicatedDataset([cf_sample])

        random_choice = random.choice([0, 1])

        clean_item = clean_dataset.__getitem__(
            0, set_character=random_choice, set_container=1 ^ random_choice
        )
        cf_item = cf_dataset.__getitem__(
            0, set_character=random_choice, set_container=random_choice
        )

        samples.append({
            "clean_prompt": clean_item["prompt"],
            "clean_ans": clean_item["target"],
            "counterfactual_prompt": cf_item["prompt"],
            "counterfactual_ans": cf_item["target"],
            "target": cf_item["target"]
        })

    return samples


def main():
    from nnsight import LanguageModel

    # Load model
    MODEL_PATH = "/net/projects/chai-lab/shared_models/Meta-Llama-3-8B-Instruct"
    print("\nLoading model...")
    model = LanguageModel(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True
    )
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded. Number of layers: {num_layers}")

    # Model evaluation
    print("\n" + "="*60)
    print("PART 1: Model Evaluation on CausalToM")
    print("="*60)

    eval_dataset = create_evaluation_samples(n_samples=20)
    correct, total = 0, 0

    for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        item = eval_dataset.__getitem__(idx)
        prompt = item["prompt"]
        target = item["target"]

        with torch.no_grad():
            with model.trace(prompt):
                pred_logits = model.lm_head.output[0, -1]
                pred_token = pred_logits.argmax(dim=-1).save()

            pred_text = model.tokenizer.decode([pred_token]).lower().strip()

            if pred_text == target.lower().strip():
                correct += 1
            total += 1

            torch.cuda.empty_cache()

    accuracy = correct / total if total > 0 else 0
    print(f"\nModel Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Pointer experiment
    print("\n" + "="*60)
    print("PART 2: Answer Lookback Pointer Experiment")
    print("="*60)

    n_samples = 20
    pointer_samples = generate_pointer_counterfactuals(n_samples)

    # Detect errors
    print("\nDetecting errors in pointer samples...")
    pointer_errors = []
    for idx, sample in tqdm(enumerate(pointer_samples), total=len(pointer_samples)):
        clean_prompt = sample["clean_prompt"]
        cf_prompt = sample["counterfactual_prompt"]
        clean_target = sample["clean_ans"]
        cf_target = sample["counterfactual_ans"]

        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(clean_prompt):
                    clean_pred = model.lm_head.output[0, -1].argmax(dim=-1).item().save()

                with tracer.invoke(cf_prompt):
                    cf_pred = model.lm_head.output[0, -1].argmax(dim=-1).item().save()

            clean_pred_text = model.tokenizer.decode([clean_pred]).lower().strip()
            cf_pred_text = model.tokenizer.decode([cf_pred]).lower().strip()

            if clean_pred_text != clean_target.lower().strip() or cf_pred_text != cf_target.lower().strip():
                pointer_errors.append(idx)

            torch.cuda.empty_cache()

    print(f"Usable pointer samples: {len(pointer_samples) - len(pointer_errors)} ({len(pointer_errors)} errors)")

    # Run pointer intervention
    pointer_layers = list(range(0, num_layers, 2))
    pointer_iia_results = {}

    print("\nRunning pointer intervention experiments...")
    for layer_idx in tqdm(pointer_layers, desc="Pointer IIA"):
        correct, total = 0, 0

        for idx, sample in enumerate(pointer_samples):
            if idx in pointer_errors:
                continue

            cf_prompt = sample["counterfactual_prompt"]
            clean_prompt = sample["clean_prompt"]
            target = sample["target"]

            with torch.no_grad():
                with model.trace(cf_prompt):
                    cf_activation = model.model.layers[layer_idx].output[0, -1].save()

                with model.trace(clean_prompt):
                    model.model.layers[layer_idx].output[0, -1] = cf_activation
                    pred = model.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_text = model.tokenizer.decode([pred]).lower().strip()
                target_text = target.lower().strip()

                if pred_text == target_text:
                    correct += 1
                total += 1

                torch.cuda.empty_cache()

        iia = correct / total if total > 0 else 0
        pointer_iia_results[layer_idx] = iia
        print(f"Layer {layer_idx}: IIA = {iia:.2f}")

    # Payload experiment
    print("\n" + "="*60)
    print("PART 3: Answer Lookback Payload Experiment")
    print("="*60)

    payload_samples = generate_payload_counterfactuals(n_samples)

    # Detect errors
    print("\nDetecting errors in payload samples...")
    payload_errors = []
    for idx, sample in tqdm(enumerate(payload_samples), total=len(payload_samples)):
        clean_prompt = sample["clean_prompt"]
        cf_prompt = sample["counterfactual_prompt"]
        clean_target = sample["clean_ans"]
        cf_target = sample["counterfactual_ans"]

        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(clean_prompt):
                    clean_pred = model.lm_head.output[0, -1].argmax(dim=-1).item().save()

                with tracer.invoke(cf_prompt):
                    cf_pred = model.lm_head.output[0, -1].argmax(dim=-1).item().save()

            clean_pred_text = model.tokenizer.decode([clean_pred]).lower().strip()
            cf_pred_text = model.tokenizer.decode([cf_pred]).lower().strip()

            if clean_pred_text != clean_target.lower().strip() or cf_pred_text != cf_target.lower().strip():
                payload_errors.append(idx)

            torch.cuda.empty_cache()

    print(f"Usable payload samples: {len(payload_samples) - len(payload_errors)} ({len(payload_errors)} errors)")

    # Run payload intervention
    payload_layers = list(range(0, num_layers, 2))
    payload_iia_results = {}

    print("\nRunning payload intervention experiments...")
    for layer_idx in tqdm(payload_layers, desc="Payload IIA"):
        correct, total = 0, 0

        for idx, sample in enumerate(payload_samples):
            if idx in payload_errors:
                continue

            cf_prompt = sample["counterfactual_prompt"]
            clean_prompt = sample["clean_prompt"]
            target = sample["target"]

            with torch.no_grad():
                with model.trace(cf_prompt):
                    cf_activation = model.model.layers[layer_idx].output[0, -1].save()

                with model.trace(clean_prompt):
                    model.model.layers[layer_idx].output[0, -1] = cf_activation
                    pred = model.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_text = model.tokenizer.decode([pred]).lower().strip()
                target_text = target.lower().strip()

                if pred_text == target_text:
                    correct += 1
                total += 1

                torch.cuda.empty_cache()

        iia = correct / total if total > 0 else 0
        payload_iia_results[layer_idx] = iia
        print(f"Layer {layer_idx}: IIA = {iia:.2f}")

    # Plot results
    print("\n" + "="*60)
    print("PART 4: Generating Plots")
    print("="*60)

    # Pointer plot
    plt.figure(figsize=(10, 4))
    layers = list(pointer_iia_results.keys())
    accuracies = list(pointer_iia_results.values())
    plt.plot(layers, accuracies, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Interchange Intervention Accuracy (IIA)', fontsize=12)
    plt.title('Answer Lookback Pointer IIA by Layer (Llama-3-8B-Instruct)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.1)
    plt.xticks(layers)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pointer_iia.png'), dpi=150)
    plt.close()
    print(f"Saved pointer_iia.png")

    # Payload plot
    plt.figure(figsize=(10, 4))
    layers = list(payload_iia_results.keys())
    accuracies = list(payload_iia_results.values())
    plt.plot(layers, accuracies, marker='o', linestyle='-', linewidth=2, markersize=6, color='green')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Interchange Intervention Accuracy (IIA)', fontsize=12)
    plt.title('Answer Lookback Payload IIA by Layer (Llama-3-8B-Instruct)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.1)
    plt.xticks(layers)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'payload_iia.png'), dpi=150)
    plt.close()
    print(f"Saved payload_iia.png")

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    layers1 = list(pointer_iia_results.keys())
    acc1 = list(pointer_iia_results.values())
    ax1.plot(layers1, acc1, marker='o', linestyle='-', linewidth=2, markersize=6, color='blue')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('IIA', fontsize=12)
    ax1.set_title('Answer Lookback Pointer', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.05, 1.1)

    ax2 = axes[1]
    layers2 = list(payload_iia_results.keys())
    acc2 = list(payload_iia_results.values())
    ax2.plot(layers2, acc2, marker='o', linestyle='-', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('IIA', fontsize=12)
    ax2.set_title('Answer Lookback Payload', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_iia.png'), dpi=150)
    plt.close()
    print(f"Saved combined_iia.png")

    # Save results
    results = {
        "model": "Meta-Llama-3-8B-Instruct",
        "num_layers": num_layers,
        "evaluation": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        },
        "pointer_experiment": {
            "n_samples": n_samples,
            "n_errors": len(pointer_errors),
            "iia_by_layer": {str(k): v for k, v in pointer_iia_results.items()},
            "peak_iia": max(pointer_iia_results.values()),
            "peak_layers": [l for l, a in pointer_iia_results.items() if a == max(pointer_iia_results.values())]
        },
        "payload_experiment": {
            "n_samples": n_samples,
            "n_errors": len(payload_errors),
            "iia_by_layer": {str(k): v for k, v in payload_iia_results.items()},
            "peak_iia": max(payload_iia_results.values()),
            "peak_layers": [l for l, a in payload_iia_results.items() if a == max(payload_iia_results.values())]
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'replication_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved replication_results.json")

    # Print summary
    print("\n" + "="*60)
    print("REPLICATION RESULTS SUMMARY")
    print("="*60)

    print(f"\nModel Used: {results['model']}")
    print(f"Number of Layers: {results['num_layers']}")

    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {results['evaluation']['accuracy']:.2%}")

    print(f"\n--- Pointer Experiment ---")
    print(f"Peak IIA: {results['pointer_experiment']['peak_iia']:.2f}")
    print(f"Peak Layers: {results['pointer_experiment']['peak_layers']}")
    print(f"Original Paper (70B): Layers 34-52")

    print(f"\n--- Payload Experiment ---")
    print(f"Peak IIA: {results['payload_experiment']['peak_iia']:.2f}")
    print(f"Peak Layers: {results['payload_experiment']['peak_layers']}")
    print(f"Original Paper (70B): After layer 56")

    print(f"\n--- Scaling Interpretation ---")
    print(f"Note: The 8B model has {results['num_layers']} layers vs 80 in 70B.")
    print(f"Proportional scaling suggests:")
    print(f"  - Pointer: layers 34-52 in 70B -> ~layers 14-21 in 8B")
    print(f"  - Payload: layers 56+ in 70B -> ~layers 22+ in 8B")

    return results


if __name__ == "__main__":
    main()
