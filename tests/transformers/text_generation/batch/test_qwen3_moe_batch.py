"""Batch tests for the Qwen3-MoE BatchLLM MXQ package release (needs four aries-rb boards)."""

MODEL_PATHS = ("mobilint/Qwen3-30B-A3B-Batch16-aries-rb",)


def test_qwen3_moe_batch(pipe, run_batch_generation) -> None:
    """Run sixteen short prompts, one per KV slot, against the batch MoE release."""
    messages = [
        [{"role": "user", "content": "Explain machine learning in one sentence."}],
        [{"role": "user", "content": "What is a function in Python?"}],
        [{"role": "user", "content": "Give one idea for a team building activity."}],
        [{"role": "user", "content": "What is the purpose of unit tests?"}],
        [{"role": "user", "content": "Write a short status update."}],
        [{"role": "user", "content": "What is an operating system?"}],
        [{"role": "user", "content": "Suggest one way to reduce meeting time."}],
        [{"role": "user", "content": "What is data preprocessing?"}],
        [{"role": "user", "content": "Name one good habit for remote work."}],
        [{"role": "user", "content": "What is recursion?"}],
        [{"role": "user", "content": "Give one tip for writing clearer prompts."}],
        [{"role": "user", "content": "What is a GPU used for?"}],
        [{"role": "user", "content": "Write a short product description."}],
        [{"role": "user", "content": "What is encryption?"}],
        [{"role": "user", "content": "Suggest one beginner coding project."}],
        [{"role": "user", "content": "What is a tokenizer?"}],
    ]
    run_batch_generation(pipe, messages)
