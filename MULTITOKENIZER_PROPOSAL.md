# Multi-Tokenizer Language Models: Shorter Sequences with Small Per-Example Output Heads

## Motivation and proposal

Vocabulary size creates an efficiency trade-off. A large vocabulary produces fewer tokens for the same text, increasing the text covered by a fixed context and reducing the number of autoregressive predictions. But a dense output layer produces \(V\) logits per predicted token: its projection costs \(O(Vd)\), normalization costs \(O(V)\), and the output table has \(O(Vd)\) parameters. This large-vocabulary output-head cost can dominate at sufficiently large \(V\), but must be measured rather than assumed to be the end-to-end bottleneck.

A small tokenizer has the inverse problem: fewer bytes per token make the same corpus longer, reduce fixed-window text coverage, and increase Transformer-body computation; the attention term grows quadratically with sequence length. We therefore propose \(K\) disjoint BPE tokenizers and a document router. The shared Transformer selects one tokenizer’s tied input/output table for a document, so it emits \(V\), rather than \(KV\), logits for that example. The stored tables still total \(KV\), comparable to a single \(KV\)-token head: this is an output-compute proposal, not a total-parameter-memory reduction. It is also distinct from the canonical *softmax bottleneck*, which is a low-rank representational limitation of linear-softmax models.

The tokenizers are discovered from unlabelled DCLM–Stack text. A balanced 100M-token sample is first collected. Matched 16K and union-vocabulary BPE teachers identify the union merges that save the most small-tokenizer pieces; these compression-residual features initialize document clusters. The procedure alternates weighted BPE fitting with soft top-\(k\) reassignment by encoding cost, prunes low-usage experts, and accepts only minimum-description-length-improving split/merge proposals. The final system has three 16,384-token experts: mainly prose, code/configuration, and structured technical text.

## Tokenizer evaluation

The discovery sample contains 62,112 documents, 100M budget tokens, and 365.4M raw bytes. Tokenizers were evaluated on unseen text: 50M bytes each from DCLM and The Stack, and their 100M-byte mixture. “Learned” uses a character-n-gram router trained to reproduce the discovered assignments; “oracle” selects the shortest expert encoding per document. Token reduction is relative to the single-16K baseline within each evaluation set.

| Evaluation text | Tokenization | Active / stored vocabulary | Tokens | Bytes/token | Tokens/KiB | Token reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DCLM + Stack, 100M bytes | Single 16K | 16K / 16K | 32,234,890 | 3.102 | 330.09 | 0.00% |
| DCLM + Stack, 100M bytes | Multi, learned router | 16K / 49K | 31,108,477 | 3.215 | 318.55 | 3.49% |
| DCLM + Stack, 100M bytes | Multi, oracle router | 16K / 49K | 31,026,627 | 3.223 | 317.71 | 3.75% |
| DCLM + Stack, 100M bytes | Single 49K | 49K / 49K | 29,175,578 | 3.428 | 298.76 | 9.49% |
| DCLM, 50M bytes | Single 16K | 16K / 16K | 12,854,921 | 3.890 | 263.27 | 0.00% |
| DCLM, 50M bytes | Multi, learned router | 16K / 49K | 12,525,319 | 3.992 | 256.52 | 2.56% |
| DCLM, 50M bytes | Multi, oracle router | 16K / 49K | 12,514,614 | 3.995 | 256.30 | 2.65% |
| DCLM, 50M bytes | Single 49K | 49K / 49K | 11,678,307 | 4.281 | 239.17 | 9.15% |
| The Stack, 50M bytes | Single 16K | 16K / 16K | 19,379,969 | 2.580 | 396.90 | 0.00% |
| The Stack, 50M bytes | Multi, learned router | 16K / 49K | 18,583,158 | 2.691 | 380.58 | 4.11% |
| The Stack, 50M bytes | Multi, oracle router | 16K / 49K | 18,512,013 | 2.701 | 379.13 | 4.48% |
| The Stack, 50M bytes | Single 49K | 49K / 49K | 17,497,271 | 2.858 | 358.34 | 9.71% |

| Router evaluation set | Documents | Learned–oracle agreement | Token regret vs. oracle | Learned expert usage (0/1/2) | Oracle expert usage (0/1/2) |
| --- | ---: | ---: | ---: | --- | --- |
| DCLM + Stack | 11,286 | 97.17% | 0.264% | 9,147 / 14 / 2,125 | 9,002 / 156 / 2,128 |
| DCLM | 9,118 | 98.33% | 0.086% | 9,045 / 2 / 71 | 8,919 / 19 / 180 |
| The Stack | 2,168 | 92.30% | 0.384% | 102 / 12 / 2,054 | 83 / 137 / 1,948 |

Router training accuracy against the discovered assignments is 96.38%. The 0.264% mixture token regret means routing accounts for only 0.26 percentage points of the mixture’s 3.75% oracle compression gain.

## Language-model cross-entropy evaluation

All LMs use the same 4-layer, 256-hidden-dimension Transformer backbone and were evaluated on the same 4,164 documents (31,818,417 UTF-8 bytes). Cross-entropy in nats/token is tokenizer-dependent and therefore not directly comparable across rows; nats/byte and bits/byte are the comparable likelihood metrics. Lower is better.

| Model / routed endpoint | Evaluated update | Documents | Predicted tokens | CE, nats/token | Perplexity | CE, nats/byte | Bits/byte |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single 16K | 9,537 | 4,164 | 10,663,555 | 3.0813 | 21.79 | 1.0327 | 1.4898 |
| Multi, all endpoints | 9,537 | 4,164 | 10,279,653 | 3.3746 | 29.21 | 1.0902 | 1.5729 |
| Multi endpoint 0 | 9,537 | 3,740 | 5,304,681 | 4.2345 | 69.02 | 1.0555 | 1.5228 |
| Multi endpoint 1 | 9,537 | 80 | 1,531,225 | 2.5168 | 12.39 | 1.4062 | 2.0286 |
| Multi endpoint 2 | 9,537 | 344 | 3,443,747 | 2.4314 | 11.37 | 1.0739 | 1.5493 |
| Single 49K | 9,000 | 4,164 | 9,721,967 | 3.2840 | 26.68 | 1.0034 | 1.4476 |

Multi-tokenization reduces evaluation sequence length by 3.60% versus single 16K (10.280M versus 10.664M predicted tokens), but it does not yet convert that gain into better likelihood: its aggregate cross-entropy is 1.5729 bits/byte, versus 1.4898 for single 16K and 1.4476 for single 49K. The endpoint breakdown identifies endpoint 1 as the principal quality problem.

## Training loss across training

Each run used the same nominal 10B-token schedule (9,537 updates; 1.05M tokens/update). The single-49K run stopped at update 9,268 (9.72B tokens). Values below are trailing means over 100 updates of the logged training cross-entropy in nats/token; the first reported point contains 47 logged updates because logging begins at update 954. These are useful optimization diagnostics but are not directly comparable across tokenizer strategies.

| Consumed training tokens | Multi: 3 × 16K | Single 16K | Single 49K |
| ---: | ---: | ---: | ---: |
| 1.05B | 5.426 | 5.124 | 5.365 |
| 2.10B | 4.713 | 4.463 | 4.668 |
| 3.15B | 4.423 | 3.717 | 3.820 |
| 4.19B | 4.196 | 3.309 | 3.503 |
| 5.24B | 3.683 | 3.147 | 3.355 |
| 6.29B | 3.438 | 3.051 | 3.249 |
| 7.34B | 3.349 | 3.075 | 3.134 |
| 8.39B | 3.279 | 2.945 | 3.196 |
| 9.44B | 2.928 | 3.046 | 3.208 |
| Final 100-update mean | 3.230 at 10.00B | 3.016 at 10.00B | 3.196 at 9.72B |

## Next step

The immediate experiment is to rerun at matched *byte* and wall-clock budgets, profile output projection separately from the Transformer body, and improve the underperforming endpoint’s data balance and routing. Success requires better bits/byte than the single-16K baseline together with a measured—not presumed—end-to-end efficiency gain.
