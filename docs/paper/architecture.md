# Materials: model architecture (MUC32_main)

The twin model is a decoder-only transformer over event tokens (`docs/paper/tokenization.md`), trained from scratch.
Every number below is read from the frozen checkpoint (`runs/train/full/MUC32_main/final/hf/config.json`,
`FROZEN.json`, code commit `fbd2ba1`) or the code that built it.

## Network

| Item | Value | Source | Note |
|---|---|---|---|
| Family | HF `Qwen3ForCausalLM`, randomly initialized | `src/icutwin/model/presets.py` | The plan's reference sizes follow CoMET (S/M/L) |
| Preset | M | `config.json` (`preset`) | The default size, compute-optimal for about 1B training tokens under CoMET's scaling fits |
| Layers | 12 | `final/hf/config.json` | All full attention (no sliding window) |
| Width (hidden size) | 768 | idem | |
| Attention heads (query / key-value) | 12 / 4 | idem | Grouped-query attention; head dimension 64 |
| Feed-forward | SwiGLU, 3,072 | idem | 4x width |
| Normalization | RMSNorm (eps 1e-6), pre-norm; Qwen3's per-head q/k norms | idem | |
| Positions | RoPE, base 1,000,000 | `presets.ROPE_THETA` | Qwen3's long-context base; M_main (8k context) used 10,000 |
| Maximum positions | 33,792 | `config.json` | Context 32,768 tokens (grammar spec `windows.max_tokens`) + 1,024 margin for the goal and ICU-day clock blocks |
| Vocabulary | 25,601 tokens | idem | `docs/paper/tokenization.md` |
| Embeddings | tied input and output | idem | |
| Parameters | 123,491,328 (embedding 19.7M, body 103.8M) | `train_report.json`; `presets.n_params` | |
| Precision | bf16 compute, float32 master weights | `train.yaml` `precision` | |
| Attention kernel (training) | PyTorch varlen FlashAttention over padding-free packed windows | `src/icutwin/model/attention.py` | Each window attends causally to itself only; checked against SDPA (`pilot.attention_parity`) |

## Constrained decoding (generation and rollouts)

| Layer | What it enforces | Source | Evidence |
|---|---|---|---|
| Grammar automaton | Block order (prefix, header, goal, timeline, end), each field once, intervals only between time groups, canonical order within a group, infusion state (a dose change or stop only while running), sub-token order | `tokenization/fsm.py`, spec `configs/grammar/spec.yaml` | Accepts 100% of the 409,297 real stays (no over-constraint); random-initialized model samples 100% valid (HF 64/64, vLLM 300/300) |
| Dataset support | A twin conditioned on a dataset draws only tokens that dataset's training windows contain | `generate/support.py` | The quality pilot found INSPIRE twins with ventilator settings INSPIRE never charts, fixed by this layer |
| Budget and liveness | The end block once only it fits; every state reaches `[EOS]` within the reserve | `generate/constrain.py` | Adversarial walks reach `[EOS]` within the reserve (`tests/test_liveness.py`) |
| Header augmentation undone | The training-time 5% unknown header values removed at sampling | `constrain.HeaderAugmentation` | The pilot discriminator's strongest sign before the fix |
| Goal obligations (optional) | Required and forbidden event sets for event-type goals | `generate/engine.py` `obligations` | The model barely conditions on goal blocks (goal tokens were loss-masked); enrichment uses filtering instead |
| Interventions (what-if) | `give` events written at the cutoff (checked by the automaton); `withhold` tokens never drawn | `engine.Intervention` | Twin-world effects only (confounding by indication is inherited) |

Sampling is ancestral at temperature 1 with no top-k, top-p or repetition penalties (mode-seeking truncation would
remove tails). Samplers: the transformers reference (`GrammarProcessor`) and vLLM 0.31 (V2 model runner, one automaton
per request slot); greedy decoding is identical between them (sampler parity).

## Serving cost (RTX 4080 SUPER, 32 GB; vLLM, one engine)

| Workload | Measured | Source |
|---|---|---|
| Twin bank | 13,416 kept twins from 42,000 draws in 2.9 h (four engines sharing the GPU) | `tools/server/bank_single.sh` |
| Rollouts, 7-day caps | 1,024 rollouts in 62-406 s per batch by dataset (about 3 per second); one engine is CPU-bound by the grammar masks (34% GPU) | `logs/p1_dev_L24.log` (Paper P) |
| S0 rollouts, 72-hour caps | 28,000 rollouts in 5,411 s | `docs/m10_factory.md` S0 |
