# BEAR_RL_MANIFEST.md — the full idea mixture, written down (2026-08-24)

> **ORIENTATION:** BearRL = the RL ENVIRONMENT + proof engine.
> WuBuMath = the math library (manifolds, encoders, codec math) that these
> environments observe and prove. Dependency: BearRL links WuBuMath, never
> the reverse. Full split + work division:
> WuBuMath `docs/WUBUMATH_BEAR_INTEGRATION.md`.

This document is the durable record of the 2026-08-24 session: the codec idea
mixture, the lineage, the harness philosophy, and the work queue. Nothing here
is speculative memory — every claim traces to a file in waefrebeorn repos.

## 1. Lineage (the tandem lab research, in order)

1. **Phase 1–5 encoders** (`wubuwizard/ENCODERS/`): symmetric geometric AE
   (`.wubu` latent format) → holomorphic QAE ("compress an image into 3
   Hamiltonian coefficients") → generative VQ-VAE+Conductor → HashMind (no
   backprop, Poincaré-ball hash memory) → Hamilton geodesic layers.
2. **Zephyr-HD "visual audio Kodak"** (`wubumind_codec.py`) — the FIRST
   audio-visual codec: audio→image reversibly (5 perceptual bands → RGB canvas,
   ISTFT decode). Proved audio and image are one representation.
3. **VHF Canvas** (`AUDIO/wubusynth/vhf_tool.py`, "solved video and audio in
   one morning"): latent field shaped like VGA — 525×656; video in 480 visible
   lines; **audio in 45 VBI lines + 16 HBI columns**; HamiltonEncoder emits
   quaternion(4)+amplitude(1) per cell; VHFDecoder = coordinate-addressable
   implicit field (bilinear sample + sin/cos positional encoding + MLP).
   Ported to C11 as `wubu_canvas` (research/066) and WuBuMath
   (`src/model/wubu_vhf_*.c`, `test_vhf_engine.c`).
4. **WuBuMath C11 slerm**: JAX core 35/35 green; quaternion/SO3/Lorentz/
   Poincaré/nested-hyperbolic wing complete; tangent-flow trainer exists
   (`src/train/wubu_tangent_flow.c`); flow matching stubbed but tested.
5. **GAAD-WuBu-ST2** (`THEORY/papers/GAAD-WuBu-ST2.md`): recursive golden
   subdivision + phi-spiral patching feeding nested hyperbolic stacks with
   SO(n) rotations — the φ front-end and the manifold backbone.
6. **Bear** (`WuBuOS/src/bear/`): full C11 RL stack — PPO/GAE/MinGRU,
   n-pole envs, Vulkan compute shaders, curriculum training, trained `.bear`
   checkpoints. Work halted inside the OS; extracted here as BearRL.

## 2. The new synthesis (2026-08-24 directive)

### 2.1 Beam-sweep canvas (memory compression)
The VHF canvas stops being a 2D framebuffer. It is a **beam**: a narrow strip
(~4000 wide) swept across content over time — CRT-style. Consequences:

- An 8K/16K frame never materializes; only the strip exists at any instant.
  Memory need is O(strip), not O(resolution).
- Orientation (horizontal vs vertical sweep) is a flag, not a re-encode —
  content is beam-based, so rotation is free.
- Audio HBI/VBI segments become reserved sweep intervals — the audio sideband
  survives by construction, and stays *invisible* to video-only consumers.

### 2.2 Golden-ratio fractal sampling (resolution agnosticism)
Apply GAAD's recursive golden subdivision ALONG the beam axis:

- Resolution becomes a **sampling depth**, not a canvas property.
- Any resolution decodes from the same latent field by querying coordinates:
  1080p = shallow subdivision resolved, 8K = deep. Beyond-Nyquist resolutions
  come free (the implicit field interpolates).
- φ-ordering preserves spatial locality in beam-time (the fractal sweep is
  load-bearing: it replaces the Hilbert-curve role). Without it, a raster beam
  destroys the locality codecs exploit.
- Compression = quantizing field cells via the Escha precision ladder.

### 2.3 Flow matching in quaternion space (the new math)
VHF is an I-frame-only engine (no P-frames). Instead of block motion vectors
(meaningless for implicit neural fields), invent P-frames as **learned geodesic
flow on the quaternion latent manifold**:

- P-frame construction: sample noise, integrate a learned probability-flow ODE
  from frame-t's latent to frame-t+1's latent along the SO(3)/hyperbolic
  geodesic; transmit only the residual.
- This is superior to Gaussian-space splatting/diffusion because the latents
  already LIVE on curved manifolds (WuBu Nesting); Euclidean noise transport
  fights that geometry, Riemannian flow rides it.
- Hard parts (honest): parallel transport between integration steps or the
  vector field drifts off-manifold (`wubu_parallel_transport.c` exists;
  wiring into probability-flow ODEs is genuinely new math).

### 2.4 Audio codec as resolution cheat (the fidelity trainer)
The kicker: audio reconstruction error is cheap to measure exactly (waveform
correlation — no perceptual video judge needed).

- Audio sideband acts as high-fidelity supervision anchoring the flow fields.
- Invisible P-frames hold audio AND train video fidelity — the audio pain-in-
  the-ass becomes the tuning signal that hells the video codec into shape.
- Zephyr-HD proved audio↔image equivalence; this reuses that proof.

### 2.5 Where BearRL fits
Every mathematical claim above becomes an RL environment:

| Claim | Env / reward |
|---|---|
| φ-fractal sweep preserves locality | reward = k-NN recall under sweep order |
| Quaternion flow hits target latents | reward = negative geodesic distance |
| Round-trip fidelity bounds | reward = PSNR/audio-corr above threshold |
| Parallel transport correctness | property harness (holonomy check) |
| Escha-ladder quantization Pareto | multi-objective rate-distortion env |

Solved-env logs = proof certificates (committed, no verbal results).

## 3. Testing harness philosophy (doctrine)

1. **Deterministic unit gate** — `make test_<name>` per module; green exit code
   or the claim doesn't exist. The test IS the spec.
2. **Property harness** — declared invariants run under fuzz seeds
   (arena alignment, GAE zero-mean advantages, bounded policy-gradient norms,
   STFT↔ISTFT round-trip correlation thresholds, exp/log map round trips on
   manifolds). A violated invariant is a bug even with all units green.
3. **Reward-as-proof** — math claims encoded as environments; a solved run with
   committed logs/artifacts is the certificate.
4. **Cross-repo parity gates** — anything BearRL proves about WuBuMath
   components must run against WuBuMath's own tests (no divergent truths).
5. Strict C11 · no monoliths · opaque struct seams · standalone test targets ·
   CPU soft fallback for all GPU paths.

## 4. Visible light / invisible light — dual-band encoding (2026-08-24)

The canvas already has this shape: visible lines = video, VBI/HBI = audio.
Generalize it into a **two-spectrum frame**:

- **Visible band** ("light"): the actual pixels the decoder renders.
- **Invisible band** ("infrared"): a parallel sweep carrying audio + video
  *processing* data — flow-matching conditioning, P-frame residuals, motion
  geodesics, sideband metadata. Never rendered; only read by the codec.

Because both bands share the SAME beam sweep and coordinate space, double
encoding costs **no extra canvas** — invisible data occupies coordinates the
renderer ignores (the HBI trick generalized to everything). The audio-sideband
fidelity trainer (§2.4) becomes one instance of this: infrared carries audio,
and its reconstruction error supervises the visible band's flow fields.

Extension idea: further bands beyond infrared (UV = provenance/watermark,
etc.) — each is just more reserved sweep segments. The frame format scales
spectrally, not spatially.

## 5. The brain moment — manifold CLIP

Status check, honestly stated: audio encoded ✅ (Kodak/VHF), video encoded ✅
(Hamilton encoder), flow-matched ideas in progress ✅ (tangent-flow trainer).
But there is no CLIP yet — and CLIP is the piece that makes modalities
*address each other*.

What CLIP actually is, stripped down: **2D-array linear algebra** — two
projection matrices W_img, W_txt mapping into a shared ℝⁿ, one cosine
similarity matrix, InfoNCE. That's the whole trick. It works because flat
linear algebra is cheap — not because flat space is right.

The WuBu version replaces every ingredient with its manifold counterpart
(all components exist in WuBuMath C11):

| CLIP ingredient | Manifold version | Existing C11 |
|---|---|---|
| Embedding space ℝⁿ | Nested hyperbolic balls `H^n_{c,s}` (WuBu Nesting) | `wubu_hyperbolic.c`, `wubu_lorentz_poincare.c` |
| Projection matrix W | Exp/Log map pair per modality (log → tangent → exp) | `wubu_manifold_ad.c` |
| Cosine similarity | Geodesic distance on Poincaré ball (+ quaternion inner product for rotation-aware pairs) | `wubu_poincare_geom.c`, `wubu_quaternion_ops.c` |
| Inter-level transition | SO(n) rotations R_i between levels | `wubu_so3.c` |
| Contrastive gradient step | Riemannian SGD on curvature c_i and scale s_i too | `wubu_riemannian_sgd.c` |
| Modality pairing | Audio↔image already proven reversible (Zephyr-HD); text joins via chunk embeddings | `wubumind_codec.py` lineage |

Why it should WIN, not just differ: hierarchy is where hyperbolic space beats
Euclidean outright. "A dog" vs "a photo of a dog running in snow" is a
*tree* of concepts; CLIP crams that tree into flat vectors and pays distortion.
Hyperbolic contrastive learning embeds trees at exponentially lower distortion
(Nickel-Kiela lineage). Nobody ships a production multimodal encoder on nested
hyperbolic manifolds. This is the gap.

Training signal comes free from the existing stack: image-text pairs from the
footage pipeline, audio-image pairs from the Kodak round-trip, and BearRL can
run the contrastive objective as an environment (reward = retrieval accuracy)
per the reward-as-proof doctrine.

"Calculus 3 LLM" note: not built yet — the nest_gpt slot exists
(`wubu_nest_gpt.c`) but untrained. The manifold-CLIP embedding layer is its
correct front door when we get there: language enters through the same
hyperbolic nesting instead of a separate token head.

## 6. Work queue

1. [x] Extract src/bear → BearRL repo
2. [ ] Build lib + revive unit targets standalone (drop WuBuOS mk deps)
3. [ ] Property harness skeleton (fuzz seeds, invariant registry)
4. [ ] First math env: quaternion geodesic reacher (tests flow-matching core)
5. [ ] Wire WuBuMath wubu_canvas as an observation space (canvas = obs)
6. [ ] φ-sweep locality env → certificate
7. [ ] Audio-sideband fidelity env → certificate
8. [ ] Dual-band frame format: visible + infrared sweep segments (spec + round-trip test)
9. [ ] **Manifold CLIP**: hyperbolic contrastive encoder — geodesic-similarity InfoNCE, Riemannian SGD on c_i/s_i
10. [ ] Manifold-CLIP retrieval env in BearRL (reward = recall@k) → certificate
11. [ ] nest_gpt front door: language enters through the manifold embedding
