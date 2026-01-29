# Do Not Waste Your Rollouts: Recycling Search Experience \\ for Efficient Test-Time Scaling

## Project Structure

```text
rse_icml/
├── code/
│   ├── standard_sampling.py                  # Baseline inference script
│   ├── experience_distillation.py            # Generates reflections (Propositions & Pitfalls)
│   ├── experience_dedup.py                   # Deduplicates reflections using embeddings
│   └── experience_guided_search.py           # Inference augmented with the Experience Bank
├── scripts/
│   └── run.sh                                # Main pipeline driver script
└── requirements.txt                          # Python dependencies
```

## Installation

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    Ensure you have Python 3.8+ and CUDA environments set up.

    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies include: `vllm`, `sentence-transformers`, `torch`, `tqdm`.*

## Usage

### Configuration

Before running the pipeline, you **must** configure the paths in `scripts/run.sh`. Open the file and update the following placeholders:

*   `MODEL_NAME`: Path to your LLM (e.g., Qwen-Thinking).
*   `QUESTION_FILE`: Path to the input dataset (JSONL format).
*   `STEP_1_ANSWER_DIR`: Path to the initial baseline results.
*   Output directories (`/path/to/output/...`).
*   Embedding model path (`/path/to/embedding/model`).

### Running the Pipeline

The `run.sh` script executes the full iterative loop (Experience Distillation -> Deduplication -> Guided Answering).

```bash
# Usage: bash scripts/run.sh <start_index> <end_index>
bash scripts/run.sh 0 100
```

### Pipeline Steps Explained

1.  **Experience Distillation**:
    The model analyzes previous attempts (from Step 1 or previous iterations) to identify:
    *   **Verified Propositions**: Irrefutable truths and intermediate conclusions (Positive Experiences).
    *   **Critical Pitfalls**: Logical fallacies and dead ends (Negative Experiences).

2.  **Deduplication**:
    Experiences are clustered and deduplicated using semantic embeddings to ensure the prompt context remains concise and diverse.

3.  **Guided Reasoning**:
    The model solves the problem again, this time with the "Experience Bank" injected into the system prompt. The model is instructed to use Verified Propositions as anchors and actively avoid Critical Pitfalls.

## Data Format

Input files should be in **JSONL** format containing at least:
*   `question_id` (or `id`)
*   `question`
*   `answer` (optional, for reference)

