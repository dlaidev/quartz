---
title: "Training Gemma with PyTorch"
description: "A causal-language-model training loop with explicit label shifting, padding masks, token-weighted accumulation, evaluation, and checkpoints."
date: 2025-06-11
tags:
  - PyTorch
  - AI
  - Training
  - HuggingFace
  - Gemma
  - Model-Training
draft: false
---

A language-model training loop turns token sequences into a loss, computes gradients, and updates parameters. Most implementation errors occur at the boundaries: constructing labels, masking padding, averaging losses, and restoring training state.

This tutorial describes those boundaries for a decoder-only Gemma model. It distinguishes a small randomly initialized model used to inspect the loop from fine-tuning a pretrained checkpoint. The examples are reference code. They were not executed during this editorial pass, and this article reports no training result.

The [Transformers Gemma documentation](https://huggingface.co/docs/transformers/model_doc/gemma) describes the model classes and configuration. Gemma releases differ in architecture and tokenizer behavior. Use the configuration and tokenizer belonging to the exact checkpoint you intend to train.

## Define the task before choosing a trainer

Pretraining starts from newly initialized parameters and uses a broad corpus. Continued language-model fine-tuning starts from a pretrained checkpoint and changes its behavior using another corpus. Supervised instruction tuning uses formatted prompt-response examples.

All three can use a causal next-token objective. Their data, compute requirements, and evaluation criteria differ.

For a token sequence $x_0,\ldots,x_{T-1}$, the basic loss is

$$
\mathcal{L}=-\frac{1}{N}\sum_{t=1}^{T-1}
 w_t\log p_\theta(x_t\mid x_0,\ldots,x_{t-1}),
\qquad N=\sum_{t=1}^{T-1}w_t.
$$

Here $w_t$ is one for a supervised target and zero for a target excluded from loss. In ordinary document training, valid tokens after the first token are supervised. For response-only instruction tuning, prompt tokens can be excluded while remaining available as input context.

The [causal-language-modeling guide](https://huggingface.co/docs/transformers/tasks/language_modeling) explains the next-token task and the standard Transformers training workflow.

## Keep inputs and targets aligned

A model returns a vocabulary-logit vector at each input position. The logits at position $t$ predict the token at position $t+1$.

For the symbolic sequence

```text
input tokens:        BOS   red   fox   EOS
prediction targets:  red   fox   EOS
logit positions:       0     1     2
```

the final input position has no following target in that sequence. The first token has no preceding logit in the supplied sequence.

Transformers causal-LM classes normally apply this shift when computing their built-in loss from `labels`. In that interface, labels usually start as an unshifted copy of `input_ids`, with excluded positions set to `-100`.

The explicit loop below computes loss itself. It does not pass `labels` to the model. It pairs `logits[:, :-1]` with `labels[:, 1:]`. Choose one convention and apply the shift once.

## Attention masks and loss masks have different jobs

A padding attention mask tells the model which input positions contain valid tokens. A loss mask determines which target positions contribute to the objective. The causal attention mask prevents a token from attending to future positions.

Setting a label to `-100` removes that target from cross-entropy. It does not hide the corresponding input token from attention. This permits response-only training: the model reads the prompt while gradients are driven by the response targets.

A collator for already-tokenized, right-padded sequences can make the two masks explicit:

```python
import torch


def collate(sequences, pad_id=0):
    if not sequences or any(len(s) < 2 for s in sequences):
        raise ValueError("each sequence needs context and a target")

    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)

    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention[row, :len(sequence)] = 1

    labels = ids.clone()
    labels[attention == 0] = -100
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}
```

This code marks padding by position. It preserves a genuine EOS target even if a tokenizer reuses its EOS token ID as the padding ID. Masking every token equal to the padding ID could remove real EOS targets in that configuration.

The example assumes right padding and one independent sequence per row. Left padding requires additional care with position IDs and loss alignment. Packing multiple documents into a row also changes which documents can attend to one another unless the attention mask enforces boundaries.

## Tokenize documents deliberately

A pretrained model expects its own vocabulary. Load the matching tokenizer and record its revision. Do not substitute token IDs from a different model.

For document fine-tuning, decide whether each example includes BOS and EOS. Check the tokenizer's special-token behavior before appending either token manually. Some tokenizers insert BOS by default and leave EOS to the caller. Duplicate special tokens change the training sequence.

Truncation is also part of the data specification. Record the maximum length and whether you discard, split, or truncate longer documents. Truncating a response can remove the useful target span. Filter examples that contain no supervised next-token target after preprocessing.

For chat data, use the checkpoint's chat template. The [chat-template guide](https://huggingface.co/docs/transformers/chat_templating) explains how role markers and control tokens become part of the token sequence. A base checkpoint and an instruction-tuned checkpoint need not accept the same format.

Response-only masking must follow the tokenized template. Splitting text at a character offset and assuming the token boundary matches can label the wrong tokens. Templates with assistant-mask support can help, but support depends on the template and library version. Inspect the rendered text and mask for representative examples.

Split training and validation data before augmentation or packing. Keep duplicates and related records in the same split when they would otherwise leak answers into validation.

## Compute a summed loss

The following function returns the summed next-token loss and the number of supervised targets:

```python
import torch.nn.functional as F


def token_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].contiguous()
    targets = batch["labels"][:, 1:].contiguous()
    count = int((targets != -100).sum().item())
    if count == 0:
        raise ValueError("batch contains no supervised target tokens")

    loss_sum = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, count
```

The model receives the complete input sequence. The loss excludes the final logit position and first label position through slicing. `logits.float()` computes the displayed cross-entropy path in float32; converting a large logits tensor can require substantial memory.

This implementation favors clarity. Production trainers may use fused losses or avoid materializing the full logits tensor. Such changes should preserve the chosen reduction and masking semantics.

## Accumulate by supervised token count

A microbatch is one forward/backward pass. An optimizer step can combine gradients from several microbatches.

If microbatch $j$ has summed loss $S_j$ and $N_j$ supervised tokens, the token-average loss over an accumulation window is

$$
\mathcal{L}_{\mathrm{window}}
=\frac{\sum_j S_j}{\sum_j N_j}.
$$

Averaging the microbatch mean losses gives each microbatch equal weight. It differs from the token average when their supervised-token counts differ. Padding and response-only masks make this distinction common.

For a single-process loop, count the window's targets first. Divide each summed loss by that common denominator before backpropagation:

```python
from itertools import islice


def windows(iterable, size):
    if size < 1:
        raise ValueError("window size must be positive")
    iterator = iter(iterable)
    while True:
        window = list(islice(iterator, size))
        if not window:
            return
        yield window


def train_epoch(model, batches, optimizer, accumulation_steps=2):
    model.train()
    epoch_sum = 0.0
    epoch_tokens = 0

    for window in windows(batches, accumulation_steps):
        total_tokens = sum(
            int((batch["labels"][:, 1:] != -100).sum().item())
            for batch in window
        )
        if total_tokens == 0:
            raise ValueError("accumulation window has no targets")

        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            loss_sum, count = token_loss(model, batch)
            if not torch.isfinite(loss_sum).item():
                raise FloatingPointError("nonfinite training loss")
            (loss_sum / total_tokens).backward()
            epoch_sum += loss_sum.detach().item()
            epoch_tokens += count

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    if epoch_tokens == 0:
        raise ValueError("training data is empty")
    return epoch_sum / epoch_tokens
```

The final, shorter window uses its actual token count. Gradients accumulate because `backward()` adds to existing parameter gradients. The loop clears gradients once per optimizer update and clips the combined gradient before that update.

The returned training mean aggregates losses measured at changing parameter values. It is useful for monitoring, but it is not an evaluation of one fixed checkpoint.

For distributed training, account for the framework's gradient averaging and the global supervised-token count. Copying this single-process normalization into a distributed loop without checking those reductions can change the effective loss scale.

## Evaluate a fixed model state

Evaluation disables training behavior and gradient recording. It must use the same label policy as training if the losses are to be compared.

```python
@torch.no_grad()
def evaluate(model, batches):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for batch in batches:
            loss_sum, count = token_loss(model, batch)
            total_loss += loss_sum.item()
            total_tokens += count
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("validation data is empty")
    return total_loss / total_tokens
```

`model.eval()` and `torch.no_grad()` serve different purposes. The first changes module behavior such as dropout. The second prevents autograd from recording the forward computation.

Perplexity is the exponential of mean next-token negative log-likelihood. Compare perplexities only when tokenization, data, and masking are compatible. A response-only loss measures a different target distribution from a loss over whole documents.

Task evaluation remains necessary. A lower validation loss can coexist with poor instruction following, incorrect answers, or degraded behavior outside the fine-tuning domain.

## Assemble a small Gemma-shaped example

The functions above can be combined with a small configuration created through Transformers:

```python
from transformers import GemmaConfig, GemmaForCausalLM

torch.manual_seed(7)
config = GemmaConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
    pad_token_id=0,
    eos_token_id=1,
    bos_token_id=2,
    use_cache=False,
)
model = GemmaForCausalLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Arbitrary token IDs exercise sequence lengths and padding.
# They are not tokenized natural-language training data.
train_batches = [
    collate([[2, 10, 11, 12, 1], [2, 13, 14, 1]]),
    collate([[2, 15, 16, 17, 18, 1]]),
    collate([[2, 19, 20, 1]]),
]
validation_batches = [collate([[2, 21, 22, 23, 1]])]

before = evaluate(model, validation_batches)
training_loss = train_epoch(model, train_batches, optimizer)
after = evaluate(model, validation_batches)
print({"validation_before": before,
       "training_mean": training_loss,
       "validation_after": after})
```

Combine `collate`, `token_loss`, `windows`, `train_epoch`, `evaluate`, and the setup block above in one Python file, including their imports. They require compatible PyTorch and Transformers installations. This configuration constructs a randomly initialized miniature model using Gemma's model class. It neither loads pretrained Gemma weights nor demonstrates useful language learning.

The small vocabulary and arbitrary IDs make the data path inspectable. The learning rate is an example setting for that miniature loop, not a recommendation for pretrained Gemma. No output values are supplied because this revised example has not been run.

Before extending it, test label alignment, padding invariance, and the final partial accumulation window. Compare accumulated gradients with a combined-batch reference under the same token-average objective. Test empty and fully masked inputs as rejected cases.

## Replace the miniature model with a checkpoint

For real fine-tuning, obtain access to the intended model under its license and download the checkpoint through an authorized workflow. Load matching model and tokenizer files. A local loading pattern is:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_dir = "/path/to/downloaded/gemma-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_dir, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_dir, local_files_only=True
)
model.config.use_cache = False
```

The path denotes a checkpoint you already possess. This loading block replaces the miniature-model setup; it is not an extra step after training that model. Construct a new optimizer from the loaded model's trainable parameters. Choose its learning rate for the fine-tuning task rather than reusing the miniature example's setting.

These calls do not establish that the model fits the available memory. Choose device placement and training precision explicitly for the environment. Move every batch tensor to the correct device before the forward pass.

Avoid treating inference-oriented automatic device placement as a complete training strategy. Full fine-tuning needs memory for trainable weights, gradients, optimizer state, activations, and temporary buffers. Loading the model for inference does not establish that backpropagation will fit.

LoRA trains added low-rank parameters while keeping base weights frozen. Quantized adapter training can reduce some memory costs, but it needs a compatible quantization backend and a training configuration designed for it. Gradient checkpointing trades recomputation for activation storage. None of these changes removes the need to validate the loss and data path.

For BF16 or FP16 training, use a suitable autocast policy. FP16 commonly requires gradient scaling; unscale gradients before clipping. Advance an update-based learning-rate schedule when an optimizer update actually occurs. A numerical overflow can cause a scaled optimizer step to be skipped.

## Save enough state to resume

Saving model weights creates an inference artifact. Resuming training also requires the optimizer state and the training position. Record scheduler and scaler state if used, along with random-generator state and data ordering.

For the single-process example, an illustrative save is:

```python
from pathlib import Path

output = Path("tiny-gemma-training-example")
if output.exists():
    raise FileExistsError(f"refusing to overwrite {output}")
output.mkdir()
model.save_pretrained(output / "model")
torch.save(
    {
        "optimizer": optimizer.state_dict(),
        "completed_epochs": 1,
        "torch_rng_state": torch.get_rng_state(),
    },
    output / "training-state.pt",
)
```

A real run must save the tokenizer, configuration, data revision, shuffle state, and any device RNG states as well. A mid-epoch checkpoint needs a data cursor. Loading only weights restarts optimization with different state.

Keep checkpoints from separate runs in separate directories. Record the model revision, precision, batch policy, token count, and software versions in a run manifest. Avoid loading serialized training state from an untrusted source.

## Read the training signals

| Symptom | First checks |
| --- | --- |
| Nonfinite loss | Inspect inputs, masks, logits, precision, and gradient norms. |
| Training loss falls while validation loss rises | Investigate overfitting, leakage, and differences between the data distributions. |
| Repeated control tokens in generated text | Inspect special-token construction and chat formatting. |
| Accumulation changes results unexpectedly | Compare supervised-token counts and loss reductions. |
| Resumed training differs from an uninterrupted run | Check optimizer state and data order before investigating nondeterministic kernels. |

A completed training experiment should report the evaluation protocol and its measured result. This article instead completes the loop's specification and provides unexecuted reference code. It makes no claim that a Gemma checkpoint was trained or improved during the revision.

## References

- [Transformers Gemma model documentation](https://huggingface.co/docs/transformers/model_doc/gemma)
- [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
- [Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [PyTorch backends](2025-05-26-torch-inductor.md): compilation is a separate step from establishing training correctness.
