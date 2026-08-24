# BearRL — the Bear RL engine, extracted and on the warpath

> Extracted from `WuBuOS/src/bear/` (work halted there, 2026-08). WuBuOS is the
> Body; BearRL is now its own module: the **math reinforcement-learning
> environment AND the proof engine** for everything WuBuMath proves moving
> forward.

## Mission

1. **RL environment** — C11 gym: envs (`bear_env*.c`), PPO trainer + GAE +
   MinGRU recurrent policies, Vulkan/CUDA backends, curriculum training.
2. **Math proof engine** — RL as a *prover*: reward = mathematical property
   verified. First client: **WuBuMath's quaternion flow-matching codec** —
   beam-sweep canvas, φ-fractal resolution-agnostic sampling, manifold P-frames,
   audio-sideband fidelity training. The agent learns/proves the codec's
   properties by interacting with it.

## What came over from WuBuOS (src/bear/, 111 files)

| Subsystem | Files | Notes |
|---|---|---|
| Arena allocator | `bear_arena.c/h` | zero-malloc core |
| Envs | `bear_env.c/h`, `bear_env_npole.c`, internal | cartpole → n-pole chain, physics |
| NN | `bear_nn.h`, `bear_nn_policy.c`, `bear_nn_value.c`, `bear_nn_ckpt.c` | forward/backward, `.bear` checkpoints |
| PPO | `bear_ppo.h`, `bear_ppo_{loss,trainer,traj}.c`, `bear_gae` shaders | full on-policy stack |
| Optimizers | `bear_opt.c/h`, `bear_holo_opt.h` | incl. holographic optimizer |
| Q-controller | `bear_qcontroller.h` | RL-tuned learning rate (same lineage as WuBuMath) |
| GPU | `bear_vulkan.{c,h}` + 20 comp shaders, `bear_cuda.h`, `bear_cudnn*`, `bear_kernels.cu` | soft fallback = CPU |
| GAAD training | `bear_gaad{,_train}.h`, `train/*gaad*` | golden-aspect curriculum experiments |
| Curriculum | `train/train_curriculum.c`, `tests/bear_train_curriculum.c` | staged difficulty |
| Trained artifacts | `policies/*.bear` | existing policy/value checkpoints |
| Math helpers | `wubu_math.h`, `wubu_warehouse.h` | shared WuBu math |

## Testing harness philosophy (the doctrine)

Every claim in this repo is proven three ways before it counts:

1. **Deterministic unit gate** — every module has a C test target; no claim of
   "works" without a green binary exit code. Same rule as WuBuMath/WuBuOffice:
   *the test IS the spec.*
2. **Property harness** — beyond units, each subsystem declares invariants
   (e.g. "GAE advantages have zero mean", "arena never returns unaligned",
   "policy gradient norm bounded"). Properties run under fuzz seeds; a violated
   invariant is a bug even if all units pass.
3. **Reward-as-proof runs** — for math claims (codec theorems, optimizer
   convergence), the RL environment encodes the claim as a reward function;
   a solved env with the recorded training log is the *certificate*. Logs are
   committed — no verbal results. (Same anti-fake-correct stance as everywhere
   else in the ~WuBu~ space: pixels/logs or it didn't happen.)

Rules: strict C11, no monolithic files, opaque structs at seams, every test
target runs standalone via `make test_<name>`.

## The idea mixture this repo will prove (2026-08-24 directive)

Full write-up: [docs/BEAR_RL_MANIFEST.md](docs/BEAR_RL_MANIFEST.md)

- VHF canvas → **beam-sweep method**: 4000-wide strip sweeps any resolution;
  orientation (horizontal/vertical) is free because content is beam-based.
- **Golden-ratio fractal subdivision along the sweep** compresses memory need;
  resolution becomes a *sampling depth*, not a canvas dimension.
- **Flow matching in quaternion space** replaces Gaussian-space splats/diffusion:
  P-frames for the I-frame-only VHF engine become geodesic ODE integrations
  between frame latents on SO(3)/hyperbolic nested manifolds.
- **Audio codec as resolution cheat**: audio lives in HBI/VBI sweep segments;
  invisible P-frames carry audio + act as high-fidelity, cheap-to-train
  supervision that tunes video fidelity.
- Resolution→resolution compression = querying the same implicit field at
  different coordinate densities (VHFDecoder already samples coordinates).
- BearRL trains/proves the vector fields and certifies reconstruction bounds.

## Build

```sh
make          # builds libbear.a
make test     # runs the unit gate
```

(GPU backends auto-detect; CPU soft fallback always available.)

## License

Umbrella License v3.0 (see LICENSE) — same as the rest of waefrebeorn/*.
