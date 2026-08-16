Nanotron implementation plan: disjoint tokenizers with a shared LLM backbone

1. Goal and recommended first version

Add a training mode in which each example is routed to exactly one tokenizer, each tokenizer has a completely disjoint local vocabulary, and the model uses a tokenizer-specific input embedding and output language-model head around one shared Transformer backbone.

The existing one-tokenizer training path must remain the default and must preserve its current configuration schema, module names, checkpoint names, and numerical behavior.

The recommended first production version is deliberately constrained:

Routing is done offline. Every document already has a stable tokenizer_id before Nanotron preprocessing starts.

A packed sequence contains tokens from one tokenizer only.

An entire optimizer step—including all gradient-accumulation microbatches—uses one tokenizer. Every data-parallel replica uses the same tokenizer for that step.

Tokenizer-local token IDs are never converted to a global union vocabulary.

Only the selected embedding/head pair is executed. The other pairs remain inactive and receive no optimizer update for that step.

The Transformer blocks, normalization layers, positional encoding, and attention/MLP parameters are shared.

Llama is implemented first. Qwen2 and Starcoder2 follow after the common endpoint abstraction and distributed tests are stable.

Online routing, per-example mixed-tokenizer batches, mid-sequence tokenizer switching, and tokenizer expert parallelism are out of scope for v1.

This restriction is not merely for convenience. Nanotron currently wraps the model with PyTorch DDP without find_unused_parameters=True, while its manual data-parallel gradient path assumes every trainable parameter has a gradient. A bank in which only one embedding/head pair is active violates both assumptions. Keeping a step homogeneous gives every replica the same active parameter set and makes the sparse-gradient behavior well-defined.

Runtime data flow

flowchart TD
    A["Offline domain assignment"] --> B["Per-tokenizer tokenized datasets"]
    B --> C["Step-level tokenizer scheduler"]
    C --> D["Selected embedding E_k"]
    D --> E["Shared Transformer backbone"]
    E --> F["Selected LM head H_k"]
    F --> G["Local-vocabulary cross entropy"]

For tokenizer (k):

[
h_0 = E_k(x),\qquad h_L = \operatorname{Backbone}(h_0),\qquad
\ell = \operatorname{CE}(H_k(h_L), y).
]

x and y contain IDs in [0, vocab_size[k]). There is no shared-token mapping and no concatenated global ID space.

2. Compatibility contract

Backward compatibility should be treated as an acceptance criterion, not as a best-effort property.

When multi_tokenizer is absent:

Parse the existing singular tokenizer: configuration exactly as today.

Construct the current singular Embedding and lm_head modules, rather than a one-element bank.

Preserve all current state-dict parameter names.

Preserve current vocabulary-padding behavior and model initialization order.

Load and resume existing checkpoints without a converter.

Do not add tokenizer-ID fields to the model call or batches in the legacy path.

Keep current Nanoset metadata and dataloader behavior.

Do not implement the feature by silently translating every existing configuration to a one-tokenizer bank. That would rename weights, change initialization/RNG consumption, complicate checkpoint loading, and make exact regression testing harder.

Use an explicit branch at configuration resolution:

if config.multi_tokenizer is None:
    runtime_vocab = SingleVocabularyRuntime.from_legacy_config(config)
else:
    runtime_vocab = MultiVocabularyRuntime.resolve(config.multi_tokenizer)

The two runtime objects can implement a common protocol, but the legacy object must instantiate the legacy modules.

3. Configuration design

3.1 Add a top-level opt-in configuration

In src/nanotron/config/config.py, retain Config.tokenizer: Optional[TokenizerArgs] and add:

@dataclass
class TokenizerSpec:
    id: int
    name: str
    tokenizer_name_or_path: str
    tokenizer_revision: Optional[str] = None
    vocab_size: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    fingerprint: Optional[str] = None

@dataclass
class TokenizerScheduleArgs:
    strategy: Literal["weighted_deficit", "round_robin"] = "weighted_deficit"
    weights: Optional[List[float]] = None
    weight_unit: Literal["tokens", "source_bytes"] = "tokens"
    seed: int = 0

@dataclass
class MultiTokenizerArgs:
    tokenizers: List[TokenizerSpec]
    schedule: TokenizerScheduleArgs
    batching: Literal["step_homogeneous"] = "step_homogeneous"
    tie_word_embeddings: bool = True
    mask_padded_vocab_logits: bool = True

@dataclass
class Config:
    tokenizer: Optional[TokenizerArgs] = None
    multi_tokenizer: Optional[MultiTokenizerArgs] = None
    # existing fields remain unchanged

The tokenizer registry should not own dataset paths. Tokenizer identity is model/checkpoint state; dataset membership is data-stage state. A training stage should refer to registry IDs.

3.2 Add a multi-tokenizer Nanoset data variant

Avoid adding parallel arrays such as dataset_folder, dataset_weights, and tokenizer_ids. Introduce structured entries:

multi_tokenizer:
  batching: step_homogeneous
  tie_word_embeddings: true
  schedule:
    strategy: weighted_deficit
    weight_unit: tokens
    seed: 42
  tokenizers:
    - id: 0
      name: web
      tokenizer_name_or_path: ./tokenizers/web
      fingerprint: sha256:...
    - id: 1
      name: code
      tokenizer_name_or_path: ./tokenizers/code
      fingerprint: sha256:...

data_stages:
  - name: stable
    start_training_step: 1
    data:
      dataset:
        type: multi_tokenizer_nanoset
        sources:
          - path: ./data/web/train
            tokenizer_id: 0
            weight: 0.70
          - path: ./data/code/train
            tokenizer_id: 1
            weight: 0.30

Keep the existing NanosetDatasetsArgs unchanged. Add a new MultiTokenizerNanosetDatasetsArgs to the relevant dataset-config union.

3.3 Resolve configuration once

Create an immutable runtime representation, preferably in a new src/nanotron/config/vocabulary.py:

@dataclass(frozen=True)
class ResolvedVocabulary:
    id: int
    name: str
    original_vocab_size: int
    padded_vocab_size: int
    special_token_ids: SpecialTokenIds
    fingerprint: str

@dataclass(frozen=True)
class ResolvedVocabularyConfig:
    mode: Literal["single", "multi"]
    vocabularies: tuple[ResolvedVocabulary, ...]
    tie_word_embeddings: bool

Pass this object explicitly from the trainer into model constructors as an optional parameter. Do not overload model_config.vocab_size with a list, and do not set it to the sum of vocabularies. The sum is a storage quantity; the active head width is a compute quantity.

Recommended constructor change:

LlamaForTraining(
    config=model_config,
    parallel_context=parallel_context,
    parallel_config=parallel_config,
    random_states=random_states,
    vocabulary_config: Optional[ResolvedVocabularyConfig] = None,
)

None means the exact current singular implementation.

3.4 Validation rules

Fail during configuration loading, before distributed initialization, if any of these checks fail:

Exactly one of tokenizer and multi_tokenizer is configured.

Multi-tokenizer mode has at least two entries.

IDs are unique, contiguous, and ordered. Names are unique.

Each resolved vocabulary is nonempty; every special-token ID is within its original vocabulary.

Each tokenizer artifact can be loaded and its measured vocabulary size matches the declared value.

Each tokenizer has an immutable fingerprint. Hash the tokenizer JSON/model, added-token table, special-token map, normalizer/pre-tokenizer settings, and pinned revision—not only the Hugging Face repository name.

Each padded vocabulary is independently divisible by the tensor-parallel size and make_vocab_size_divisible_by requirement.

Every data source names a known tokenizer ID and its dataset manifest fingerprint matches that tokenizer.

All training stages use the same ordered tokenizer registry. Stages may change mixture weights, but v1 must reject adding, removing, or reordering tokenizers during a run.

Multi-tokenizer mode only accepts step_homogeneous batching in v1.

LightEval and generation settings that assume a single tokenizer either specify a tokenizer ID or fail clearly.

4. Data preprocessing and storage

4.1 Separate routing from Nanotron training

The domain-discovery/router pipeline should output records with a stable assignment, for example:

{"text": "...", "tokenizer_id": 1, "routing_version": "domains-2026-08-15"}

Nanotron preprocessing should consume that assignment. It may load all tokenizer artifacts, but it should not retrain or update the router during LLM training. This makes tokenized data immutable, makes resume exact, and avoids a moving token-ID interpretation.

Add tools/preprocess_multi_tokenizer_data.py, initially as a wrapper around the existing preprocessing machinery:

Read routed raw documents.

Validate tokenizer_id.

Select the corresponding tokenizer.

Append the tokenizer-specific EOS/document separator.

Write the document only into that tokenizer's output shard/folder.

Pack sequences only within that folder.

Emit per-folder and top-level manifests.

Do not pack two tokenizers into one sequence, even if both have unused space. The same integer has unrelated meanings in the two vocabularies, and the model accepts only one embedding/head selection for the sequence.

4.2 Versioned dataset manifest

The current Nanoset configuration infers a single tokenizer from .metadata and asserts that all folders match it. Preserve that reader for legacy data. For multi-tokenizer data, add a JSON manifest such as:

{
  "format_version": 1,
  "tokenizer_id": 1,
  "tokenizer_name": "code",
  "tokenizer_name_or_path": "./tokenizers/code",
  "tokenizer_revision": null,
  "tokenizer_fingerprint": "sha256:...",
  "original_vocab_size": 16384,
  "token_size_in_bytes": 2,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "routing_fingerprint": "sha256:...",
  "documents": 123456,
  "tokens": 987654321,
  "source_bytes": 4012345678
}

Token storage width can differ by folder. A 16K tokenizer fits in two-byte unsigned storage, while another tokenizer may require four bytes. The loader should derive dtype from each folder's manifest.

4.3 Dataset and sampler

Add src/nanotron/data/multi_tokenizer.py with two components:

MultiTokenizerDataset: owns one existing Nanoset/TokenizedBytes dataset per tokenizer and exposes per-tokenizer lengths and consumption statistics.

HomogeneousTokenizerStepSampler: chooses one tokenizer for an optimizer step and supplies all accumulation microbatches for that tokenizer.

The sample payload remains ordinary local IDs plus a scalar tokenizer ID:

{
    "input_ids": LongTensor[sequence_length + 1],
    "tokenizer_id": int,
    "dataset_source": str,   # optional bookkeeping, not sent through model
}

The scheduler must be a pure deterministic function of:

data-stage ID,

optimizer step within the stage,

schedule seed,

normalized tokenizer weights.

A weighted-deficit scheduler is preferable to independent random draws because it limits short-window starvation:

def select_tokenizer(step, weights, seed):
    # Stateful form shown for clarity; checkpoint deficits or reconstruct them.
    deficits += weights
    k = argmax(deficits, stable_seeded_tiebreak(seed, step))
    deficits[k] -= 1.0
    return k

All DP replicas compute the same k. They then draw different sample indices from tokenizer k using the existing DP sharding rules. TP and PP ranks within a model replica see the same step metadata. In debug mode, all-gather the selected ID across the DP/CP group and assert equality before the forward pass.

Checkpoint either the deficit vector and each tokenizer's cursor, or make both exactly reconstructable from (stage, step, seed). Store per-tokenizer consumed samples/tokens even if reconstruction is possible, because it enables sanity checks and reporting.

4.4 Define mixture weights precisely

Different tokenizers encode different amounts of source data per token. Therefore a “30% code” mixture is ambiguous.

If weights describe training-token share, schedule tokenizer k with p_k = weight_k.

If weights describe source-byte share q_k, and measured compression is b_k bytes/token, use

[
p_k = \frac{q_k / b_k}{\sum_j q_j / b_j}.
]

Expose the unit in configuration and log both consumed tokens and estimated/known source bytes. Never silently interpret document weights as token weights.

5. Dataloader and pipeline interfaces

Modify src/nanotron/data/clm_collator.py only for the multi-tokenizer collator. Add two scalar fields:

input_tokenizer_id: real tensor on the input pipeline rank; TensorPointer elsewhere.

label_tokenizer_id: real tensor on the output pipeline rank; TensorPointer elsewhere.

They contain the same ID. Separate names avoid assuming that the first and last pipeline stages are colocated. With PP=1, both can reference the same scalar tensor.

The resulting model-facing batch is conceptually:

{
    "input_ids": ...,
    "input_mask": ...,
    "label_ids": ...,
    "label_mask": ...,
    "input_tokenizer_id": tensor(k),
    "label_tokenizer_id": tensor(k),
}

Requirements:

Validate that every example collated into a microbatch has the same ID.

Validate that every microbatch in the optimizer step has the scheduled ID.

Treat the ID as replicated metadata under context parallelism; do not slice it along the sequence dimension.

Preserve the current collator and forward signature in single-tokenizer mode.

In run_train.py, replace the singular AutoTokenizer load/assertion only in the multi-tokenizer data branch. Load the registry and validate each source manifest independently.

6. Model endpoint banks

6.1 Common modules

Add src/nanotron/models/multi_tokenizer.py:

class DisjointEmbeddingBank(nn.Module):
    embeddings: nn.ModuleList  # TensorParallelEmbedding per tokenizer

    def forward(self, input_ids, tokenizer_id):
        k = checked_scalar_id(tokenizer_id)
        return self.embeddings[k](input_ids)

class DisjointLMHeadBank(nn.Module):
    heads: nn.ModuleList  # TensorParallelColumnLinear per tokenizer

    def forward(self, hidden_states, tokenizer_id):
        k = checked_scalar_id(tokenizer_id)
        logits = self.heads[k](hidden_states)
        return mask_padding_columns(logits, k)

Each embedding maps its local vocabulary to the same hidden size. Each head maps the shared hidden size back to its own local vocabulary. Do not compute all heads and select afterward; that would erase the main compute benefit.

Use ModuleList, not a dictionary keyed by mutable names, so parameter ordering and checkpoint names are deterministic. Stable names should look like:

model.token_position_embeddings.pp_block.token_embeddings.embeddings.0.weight
model.lm_heads.0.weight
model.token_position_embeddings.pp_block.token_embeddings.embeddings.1.weight
model.lm_heads.1.weight

The exact prefix should follow the existing Llama module tree; the important property is stable numeric ID ordering.

6.2 Modify Llama without disturbing the legacy branch

In src/nanotron/models/llama.py:

Keep the existing Embedding class unchanged.

In LlamaModel.__init__, instantiate current singular modules when vocabulary_config is None or mode == "single".

In multi mode, instantiate DisjointEmbeddingBank and DisjointLMHeadBank using independently padded vocabulary sizes.

Add tokenizer-ID keys to the first and last PipelineBlock input specifications only in multi mode.

The decoder block list and final normalization are identical in both modes.

LlamaForTraining.forward accepts the two tokenizer-ID arguments only in the multi path and validates their equality when PP=1.

Reuse the existing sharded cross-entropy implementation on the selected head's logits and local labels.

Avoid a single union-sized embedding/head with offsets. That would technically keep vocabularies disjoint, but it would make every output softmax pay for the sum of all vocabulary sizes and would not realize the intended active-vocabulary compute reduction.

6.3 Vocabulary padding

Track both original_vocab_size[k] and padded_vocab_size[k]. Token IDs must remain below the original size. If tensor parallelism adds output rows, those rows must not be valid softmax classes.

For multi mode, mask every local logit column whose global vocabulary index is at least the original size to negative infinity before the sharded cross entropy. Each TP rank can compute its global column offset. Add tests for cases where the padding lives entirely on the last shard and where it crosses a shard boundary.

Leave the legacy path's current behavior unchanged; changing existing vocabulary-padding semantics belongs in a separate fix.

6.4 Per-tokenizer weight tying

Nanotron's current get_embeddings_lm_head_tied_names() returns one flat group. Returning all bank weights in that list would incorrectly tie every tokenizer together.

Add a plural API to the base model contract:

def get_embeddings_lm_head_tied_groups(self) -> list[list[str]]:
    legacy = self.get_embeddings_lm_head_tied_names()
    return [legacy] if legacy else []

Update mark_tied_parameters() in src/nanotron/trainer.py to call tie_parameters() once per group. The legacy adapter produces exactly one group and preserves current behavior. Multi-tokenizer Llama returns:

[
    [embedding_name(0), head_name(0)],
    [embedding_name(1), head_name(1)],
]

Never tie weights across tokenizer IDs, even if two tokenizers happen to have the same vocabulary size.

6.5 Initialization, parameter counts, and FLOPs

Initialize every bank member with the existing embedding/head initialization policy.

Use deterministic per-tokenizer RNG names derived from stable IDs so adding logging or changing iteration order cannot perturb initialization.

Report shared_parameters, endpoint_parameters_total, and parameters_active_per_step separately.

Memory accounting uses the sum of all endpoint parameters and optimizer states.

Step FLOPs use only the selected head size. Expected-FLOP reporting can use the schedule-weighted average active vocabulary; pipeline partitioning should use the maximum active endpoint cost for safety.

Test spectral/μP initialization explicitly; fan-in/fan-out calculations must be per head, not based on summed vocabularies.

7. Distributed training and sparse endpoint gradients

This is the highest-risk part of the change.

7.1 Step-homogeneous active set

For v1, choose one tokenizer per optimizer step, before the pipeline engine consumes its microbatches. Store it in a StepRoutingContext passed to the dataloader/collator and used for gradient synchronization.

The active set for step s is:

all shared backbone parameters
+ embedding parameters for k(s)
+ LM-head parameters for k(s)

If input and output weights are tied, the tied group for k(s) is active; every other tied group is inactive.

7.2 Tag endpoint parameters

During model construction, attach stable metadata to every endpoint parameter, for example:

param.set_metadata("tokenizer_endpoint_id", k)
param.set_metadata("tokenizer_endpoint_kind", "embedding" | "lm_head")

Prefer extending NanotronParameter metadata rather than relying on parameter-name parsing. Shared parameters have no endpoint ID.

7.3 PyTorch DDP path

In DistributedTrainer._init_model, set find_unused_parameters=True only when multi-tokenizer mode is enabled. Keep the current constructor arguments exactly unchanged in single mode.

Because an optimizer step is tokenizer-homogeneous, the unused set is stable across its accumulation microbatches. Verify compatibility with both pipeline engines and Nanotron's no_sync/accumulation behavior. If DDP's unused-parameter traversal proves incompatible with pipeline execution, use the explicit active-set reducer described below instead of forcing dummy zero-valued head computations.

Do not “touch” every inactive head with 0 * parameter.sum() merely to satisfy DDP. It creates autograd and communication overhead, may cause AdamW to decay inactive heads, and conceals synchronization bugs.

7.4 Manual DP, FP32 accumulation, and ZeRO-1 path

Refactor the current unconditional gradient loop in src/nanotron/parallel/data_parallel/utils.py into an active-aware reducer:

for name, param in sorted(model.named_parameters()):
    if is_inactive_endpoint(param, active_tokenizer_id):
        assert param.grad is None or accumulator.is_pristine(name)
        continue
    grad = accumulator.get_grad(name) if accumulator else param.grad
    assert grad is not None
    all_reduce(grad, group=dp_group)

All ranks must build the same ordered collective list. Assert the active tokenizer ID across the DP group before entering this loop.

Update src/nanotron/optim/gradient_accumulator.py so a preallocated zero buffer is not mistaken for a real accumulated gradient. Track a has_grad bit per parameter for the current optimizer step. Before optimizer.step():

expose gradients only for shared parameters and the active endpoint;

set inactive endpoint gradients to None;

do not advance inactive endpoint Adam moments or parameter-local step counters;

do not apply weight decay to inactive endpoints.

This gives each endpoint an optimizer time scale based on the number of steps in which it was trained. Log both global optimizer steps and per-tokenizer active steps.

For ZeRO-1, every DP rank must make the same active/inactive decision even if only one rank owns a given optimizer-state shard. Add a distributed assertion before stepping and a test that inactive shards remain byte-identical.

7.5 Tied-gradient synchronization

sync_tied_weights_gradients() currently obtains the gradient or accumulator buffer for every tied parameter group. Extend it with the active tokenizer ID and skip inactive tokenizer groups in a deterministic order. For the active group, require a real gradient on both pipeline endpoints before the reduction.

7.6 Future microbatch-level routing

After v1 is stable, different accumulation microbatches may use different tokenizers. That requires an active_tokenizers_this_step bitset, careful DDP hook behavior across multiple forwards, and optimizer updates for every endpoint used at least once. It should be a separately tested feature flag, not an implicit relaxation of step_homogeneous.

8. Checkpointing and exact resume

Nanotron's generic weight serializer should naturally enumerate bank parameters, but multi-tokenizer identity cannot be inferred safely from tensor shapes or names alone.

8.1 Checkpoint manifest

Write multi_tokenizer_manifest.json at checkpoint root on world rank zero:

{
  "format_version": 1,
  "ordered_tokenizers": [
    {
      "id": 0,
      "name": "web",
      "fingerprint": "sha256:...",
      "original_vocab_size": 16384,
      "padded_vocab_size": 16384,
      "special_token_ids": {"bos": 1, "eos": 2, "pad": 0}
    }
  ],
  "tie_word_embeddings": true,
  "batching": "step_homogeneous",
  "schedule_version": 1,
  "routing_fingerprint": "sha256:..."
}

Also save immutable tokenizer artifacts in or adjacent to the checkpoint, or record pinned revisions plus content fingerprints. A mutable repository path is insufficient for reproducibility.

8.2 Extend training metadata compatibly

The current DataStageMetadata already stores aggregate samples and consumed tokens per dataset folder. Add optional fields with defaults so old checkpoints still deserialize:

consumed_samples_per_tokenizer: Dict[str, int] = field(default_factory=dict)
consumed_tokens_per_tokenizer: Dict[str, int] = field(default_factory=dict)
active_steps_per_tokenizer: Dict[str, int] = field(default_factory=dict)
tokenizer_scheduler_state: Optional[Dict[str, Any]] = None

Prefer string keys in serialized JSON for stable compatibility. On resume, cross-check these totals against folder consumption statistics.

8.3 Resume validation

Before loading weights or optimizer state, require an exact match on:

tokenizer count and ordered IDs;

tokenizer fingerprints;

original and padded vocabulary sizes;

special-token IDs;

per-tokenizer tying policy;

schedule algorithm/version and data-stage tokenizer membership.

Reject reordered tokenizers even if all tensor shapes match. Otherwise embeddings.0 could silently change semantic meaning.

Restoring a checkpoint must reproduce the next tokenizer ID, the next per-tokenizer sample indices, and all existing Nanotron RNG streams exactly.

8.4 Explicit conversion utilities

Do not put heuristic conversion in normal checkpoint loading. Add explicit tools later:

convert_single_to_multi.py: load the shared backbone; copy the old embedding/head only into a tokenizer with the exact same fingerprint; randomly initialize other endpoints; reset optimizer state for new endpoints.

extract_tokenizer_endpoint.py: select tokenizer k plus the shared backbone and emit a single-tokenizer checkpoint.

Adding/removing tokenizers in-place, resizing a tokenizer vocabulary, and reordering IDs are unsupported in v1.

9. Logging and observability

Add the following metrics:

tokenizer/active_id and tokenizer/active_name;

tokenizer/<name>/active_steps;

tokenizer/<name>/consumed_samples;

tokenizer/<name>/consumed_tokens;

tokenizer/<name>/source_bytes when present in manifests;

tokenizer/<name>/loss and optional z-loss;

tokenizer/<name>/grad_norm_embedding and grad_norm_head at a low logging frequency;

tokenizer/<name>/tokens_per_second;

schedule target share versus realized share;

shared-backbone grad norm separately from endpoint grad norms.

The headline training loss should be token-weighted across observations, not a naive average of per-tokenizer averages. Keep per-tokenizer curves, because a global curve can hide a failing or starved endpoint.

Warn if a tokenizer has not been scheduled within a configurable number of steps. A weighted-deficit scheduler should make this bound predictable.

10. Generation, evaluation, and export

These can follow training support, but the interfaces should be decided early.

For generation:

Require tokenizer_id or invoke an external router before tokenization.

Use that tokenizer for the prompt, selected embedding/head, and decoding.

Store the tokenizer ID with the KV cache and reject a mid-sequence change.

Group batch-generation requests by tokenizer ID.

For evaluation:

Run perplexity/loss datasets with an explicit tokenizer ID and report per-tokenizer metrics.

Do not let LightEval silently use tokenizer zero. Initially, reject multi-tokenizer LightEval configurations unless a tokenizer ID is specified or the runner is extended to group tasks by tokenizer.

Compare losses in bits/byte or nats/byte as well as loss/token when comparing tokenizers, because token units differ.

For export:

A standard single-vocabulary Transformers checkpoint cannot represent multiple disjoint embeddings/heads without custom model code.

Support either a custom multi-tokenizer model export or one single-tokenizer export per selected endpoint.

11. Concrete file-by-file change map

Area

Files

Planned change

Config

src/nanotron/config/config.py

Add opt-in registry/schedule/data args; keep singular args unchanged; validate exclusivity.

Runtime vocab

src/nanotron/config/vocabulary.py (new)

Resolve sizes, TP padding, fingerprints, special IDs, and mode.

Model configs

src/nanotron/config/models_config.py

Avoid union vocabulary fields; only add hooks needed to pass resolved runtime config.

Preprocessing

tools/preprocess_multi_tokenizer_data.py (new)

Route to tokenizer-specific writers; prohibit cross-tokenizer packing; emit manifests.

Data

src/nanotron/data/multi_tokenizer.py (new)

Dataset registry, deterministic step scheduler, per-tokenizer cursors/stats.

Existing loaders

src/nanotron/data/nanoset.py, src/nanotron/data/tokenized_bytes.py

Add manifest-aware construction without changing legacy readers.

Collation

src/nanotron/data/clm_collator.py

New multi-tokenizer collator and PP tokenizer-ID tensors.

Dataloader build

src/nanotron/data/dataloader_builder.py, run_train.py

Dispatch new dataset type; resolve all tokenizers; set step routing context.

Common model code

src/nanotron/models/multi_tokenizer.py (new)

Disjoint TP embedding/head banks, ID validation, padded-logit masking.

Llama

src/nanotron/models/llama.py

Multi-mode endpoint branch and pipeline keys; exact legacy branch retained.

Other models

src/nanotron/models/qwen.py, src/nanotron/models/starcoder2.py

Adopt common bank after Llama validation.

Trainer

src/nanotron/trainer.py

Pass runtime vocab, select ID per step, DDP unused-param mode, plural tie groups, metrics.

DP sync

src/nanotron/parallel/data_parallel/utils.py

Active-aware ordered reductions and assertions.

Accumulation

src/nanotron/optim/gradient_accumulator.py

Real-grad bits; inactive endpoint gradients remain None.

Optimizer/ZeRO

optimizer factory and ZeRO state code

Skip inactive endpoint params without advancing moments or applying decay.

Tied weights

src/nanotron/parallel/tied_parameters.py

Skip inactive tie groups; support multiple independent endpoint pairs.

Metadata

src/nanotron/serialize/metadata.py and checkpoint save/load

Optional per-tokenizer counters/state and manifest validation.

Generation/eval

run_generate.py, generation helpers, LightEval adapter

Explicit routing, per-tokenizer decoding/evaluation, clear unsupported errors.

Examples/docs

examples/, docs/

Two-tokenizer config, preprocessing guide, invariants, checkpoint conversion.

Verify exact filenames against the target Nanotron commit before opening PRs; the repository is actively changing.

12. Implementation sequence

PR 1 — Lock down the legacy baseline

Add snapshot tests for representative single-tokenizer configs.

Record state-dict keys for tied and untied Llama.

Add a tiny deterministic training fixture and checkpoint/resume trajectory.

Cover TP=1/2 and PP=1/2 where practical.

Exit condition: later PRs can prove the legacy path's config, names, logits, gradients, and resume behavior are unchanged.

PR 2 — Configuration and manifests, no model behavior yet

Add configuration dataclasses and runtime resolution.

Add tokenizer fingerprinting and dataset/checkpoint manifest schemas.

Add validation and YAML round-trip tests.

Reject multi mode with a clear “not yet enabled” error after validation.

Exit condition: invalid combinations fail before distributed startup; old configs round-trip unchanged.

PR 3 — Preprocessing and homogeneous data scheduling

Add routed preprocessing, per-tokenizer writers, manifests, and stats.

Add MultiTokenizerDataset and deterministic step scheduler.

Add new collator fields and PP/CP handling.

Add resumeable per-tokenizer cursors.

Exit condition: a CPU-only test can enumerate batches, prove no mixed IDs, reproduce the sequence after resume, and hit target mixture shares.

PR 4 — Llama endpoint banks on one GPU

Add disjoint embedding/head banks.

Wire tokenizer IDs through Llama pipeline blocks.

Add local-vocabulary cross entropy and padded-column masking.

Add independent tying groups.

Add endpoint-aware initialization/count/FLOP reporting.

Exit condition: two-tokenizer Llama trains on synthetic data with DP=TP=PP=1; only the selected endpoint changes each step.

PR 5 — Distributed sparse-gradient correctness

Add step routing context and global ID assertions.

Enable DDP unused-parameter handling only for multi mode.

Make manual DP, gradient accumulation, tied reduction, clipping, and ZeRO-1 active-aware.

Add inactive optimizer-state tests.

Exit condition: the distributed matrix below runs without a hang or silent inactive-head update.

PR 6 — Checkpoint, resume, and conversion

Save/load manifests and per-tokenizer scheduler/cursor metadata.

Validate identities before tensor load.

Test exact resume across data stages.

Add explicit single-to-multi and endpoint-extraction utilities.

Exit condition: resumed runs reproduce the next IDs, samples, losses, and parameter updates.

PR 7 — Performance and operational hardening

Benchmark throughput/MFU and memory.

Verify only the active head kernel runs.

Add detailed metrics and starvation warnings.

Run long multi-node soak tests.

Exit condition: active-vocabulary compute scales with V_k, not sum(V_k), and overhead versus an equivalent single active vocabulary is within the agreed budget.

PR 8 — Other architectures, generation, and evaluation

Apply the common endpoint abstraction to Qwen2 and Starcoder2.

Add explicit-tokenizer generation and per-tokenizer evaluation.

Document composition with a Transformer MoE backbone.

13. Test matrix

13.1 Unit and property tests

Legacy config parses and serializes without new keys.

Multi config rejects duplicate/reordered IDs, fingerprint mismatches, bad special IDs, and unknown dataset IDs.

Scheduler is deterministic, respects weights, bounds starvation, and resumes exactly.

Every batch and optimizer step is tokenizer-homogeneous.

Context parallelism does not slice tokenizer IDs.

Bank dispatch selects exactly one module and rejects nonscalar/out-of-range IDs.

Variable vocabulary sizes work; padded classes have zero probability contribution.

Each tied embedding/head pair shares/synchronizes only with its matching ID.

Inactive gradients are None; inactive weights and optimizer state remain byte-identical after a step.

13.2 Numerical equivalence tests

Legacy single-tokenizer mode before versus after the change: identical state-dict keys and, under deterministic kernels, identical logits/loss/gradients.

Copy a single-tokenizer embedding/head into both endpoints of a two-tokenizer model and feed identical local IDs. Routing to either endpoint should match the single model.

Tied and untied variants should match their corresponding reference calculations.

Sharded cross entropy with a selected head should match dense PyTorch cross entropy on a tiny model.

13.3 Distributed matrix

At minimum cover:

Dimension

Values

DP

1, 2

TP

1, 2

PP

1, 2

CP

1, 2 where currently supported

ZeRO

0, 1

Gradient accumulation

1, 2, greater than tokenizer count

Embedding/head tying

off, per-tokenizer on

Pipeline engine

AFAB, 1F1B

Vocabulary sizes

equal, unequal, TP-padded

For every combination that Nanotron itself supports, assert:

no collective hang;

identical active tokenizer across DP replicas;

shared gradients agree across replicas;

the active endpoint updates;

inactive endpoints, Adam moments, and weight decay state do not change;

tied gradients agree across the relevant PP ranks;

checkpoint/resume selects the same next tokenizer and samples.

13.4 End-to-end tests

Two synthetic domains with disjoint alphabets; both per-tokenizer losses must decrease.

A small web/code corpus run using the learned assignments and tokenizers from the compression experiment.

At least a 1,000-step multi-node soak test with checkpoint/resume in the middle.

Data-stage transition with changed mixture weights but unchanged tokenizer registry.

Expected failures for registry mutation, cross-tokenizer packing, and unsupported evaluation/export.

14. Performance targets and measurements

Measure against a single-tokenizer model with the same active vocabulary size and sequence/batch configuration.

Forward/backward endpoint FLOPs should be proportional to the selected V_k, not sum(V_k).

Endpoint parameter and optimizer memory will scale with sum(V_k); report this separately from active compute.

DDP unused-parameter discovery and dynamic dispatch should add no more than roughly 2–5% step-time overhead in the initial target. Treat this as a measurement target, not a guaranteed result.

Data scheduling must not starve GPUs when one tokenizer's shards are slower or smaller. Measure dataloader wait time by tokenizer.

Log tokens/s and source bytes/s. A denser tokenizer may reduce sequence count for the same source corpus even if tokens/s is unchanged.

If endpoint memory becomes limiting, a future feature could shard tokenizer banks across an expert-parallel-like group and move hidden states to the selected endpoint. Do not do this in v1: it adds all-to-all communication and makes the design much closer to MoE routing. With two or a few 16K vocabularies, replicated endpoint banks are much simpler.

15. Risks and mitigations

Risk

Consequence

Mitigation

Different DP ranks select different tokenizer IDs

Collective mismatch, hang, or averaging unrelated heads

Pure step scheduler plus pre-forward all-gather assertion.

DDP/manual reducer assumes all params are used

Runtime error or hang

Multi-only unused-param mode and explicit active-set reducer.

Zero buffers make inactive heads look active

Adam moments/weight decay silently modify them

Track has_grad; expose None for inactive endpoints.

Flat tied-weight API ties all tokenizers together

Destroys disjoint endpoints

Plural independent tie groups with legacy adapter.

Tokenizer registry reordered on resume

Correct shapes but corrupt semantics

Ordered fingerprinted checkpoint manifest and fail-fast validation.

Padded rows enter the softmax

Probability mass assigned to nonexistent tokens

Store original size and mask padded sharded columns.

Mixture weights use ambiguous units

Domain mix differs from intent

Require tokens or source_bytes; log realized shares.

Cross-tokenizer packing

IDs are decoded by the wrong endpoint

Per-tokenizer writers and collator assertions.

Rare tokenizer gets too few updates

Poor endpoint and domain performance

Weighted-deficit scheduling, per-tokenizer active-step metrics, starvation warnings.

Standard evaluation/export assumes one tokenizer

Misleading results or invalid checkpoint

Require explicit tokenizer ID or fail; provide endpoint extraction.

Legacy behavior drifts

Existing Nanotron users break

Separate code path and baseline snapshot/equivalence tests.

16. Definition of done

The feature is ready for normal use when all of the following hold:

Existing single-tokenizer YAML files and checkpoints run without modification and preserve state-dict names.

A two-or-more-tokenizer Llama trains with fully disjoint local IDs and no global union vocabulary.

Only one embedding and one head execute per step; endpoint compute depends on the active vocabulary size.

Every supported DP/TP/PP/CP/ZeRO combination has explicit sparse-gradient coverage and no collective hangs.

Inactive endpoint weights and optimizer state remain unchanged on inactive steps.

Per-tokenizer embedding/head tying works without cross-tokenizer tying.

Exact resume reproduces routing, data cursors, RNG, and the next loss trajectory.

Dataset and checkpoint fingerprints prevent tokenizer-semantic mismatches.

Per-tokenizer losses, utilization, token counts, and byte counts are observable.

The web/code end-to-end test shows both domains learning, and a multi-node soak test completes successfully.

17. Upstream code facts informing this plan

This plan is based on the current Nanotron main branch as inspected on 2026-08-15:

Nanotron advertises DP, TP, PP, expert parallelism, ZeRO-1, tied parameters, and checkpointing, all of which the new endpoint bank must compose with: Nanotron repository.

Current Nanosets are pretokenized and their documented setup assumes one tokenizer whose vocabulary matches the model: Nanoset documentation.

Config currently has a singular tokenizer, and the Nanoset validation requires dataset folders to agree on it: configuration source.

Llama currently constructs one tensor-parallel embedding and one tensor-parallel LM head using a singular vocabulary size: Llama model source.

The trainer pads one model_config.vocab_size, constructs DDP without unused-parameter discovery, and treats the current embedding/head names as one tied group: trainer source.

Current checkpoint metadata already has data-stage and per-folder token-consumption fields, providing a natural backward-compatible location for optional per-tokenizer scheduler/cursor data: checkpoint metadata source.

Tied-parameter synchronization currently obtains gradients for all tied groups, so it must become active-group-aware: tied parameter source.