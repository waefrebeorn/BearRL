#!/usr/bin/env python3
"""
propgate.py -- BearRL/WuBuMath property-harness skeleton (GAP-F003).

Doctrine (docs/BEAR_RL_MANIFEST.md §3): a claim exists only when
  1. its unit gate is green, AND
  2. its declared invariants hold under fuzzed seeds, AND
  3. math claims carry committed reward-as-proof certificates.

This runner executes registered property checks across the repos:
  - each property = (name, repo, command, invariant description)
  - every check runs over N fuzz seeds; any failure fails the gate
  - results are emitted as JSON for the dashboard + committed logs

Usage:
  python3 tools/propgate.py [--seeds N] [--json OUT]
Exit code: 0 iff all properties hold on all seeds.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
WUBUMATH = os.path.abspath(os.path.join(HERE, "..", "..", "wubumath"))
BEARRL   = os.path.abspath(os.path.join(HERE, ".."))

# ---------------------------------------------------------------- registries
# Each property: dict(name, repo, cmd_template with {seed}, invariant)
PROPERTIES = [
    dict(
        name="quat_norm_preserved",
        repo="wubumath",
        cmd="bin/test_flow_matching",           # suite asserts on-manifold invariants
        invariant="FM trajectories stay inside Poincare ball (norm<1)",
    ),
    dict(
        name="target_velocity_tangent",
        repo="wubumath",
        cmd="bin/test_flow_matching",
        invariant="geodesic target velocity keeps paths on-manifold & toward target",
    ),
    dict(
        name="heun_solver_on_manifold",
        repo="wubumath",
        cmd="bin/test_flow_matching",
        invariant="HEUN ODE output strictly inside ball; matches EULER family",
    ),
    dict(
        name="mclip_geodesic_metric",
        repo="wubu" + "math",
        cmd="bin/test_manifold_clip",
        invariant="d(x,x)=0, symmetric, triangle inequality on ball samples",
    ),
    dict(
        name="infonce_decreases",
        repo="wubumath",
        cmd="bin/test_manifold_clip",
        invariant="contrastive loss decreases under training; recall@1 > chance",
    ),
    dict(
        name="beam_invisible_band_never_renders",
        repo="wubumath",
        cmd="bin/test_beam",
        invariant="infrared band bytes never appear in renderer output",
    ),
    dict(
        name="beam_phi_order_progressive_uniformity",
        repo="wubumath",
        cmd="bin/test_beam",
        invariant="phi-order prefixes cover sweep near-uniformly (3-distance)",
    ),
    dict(
        name="lorentz_flow_on_hyperboloid",
        repo="wubumath",
        cmd="bin/test_lorflow",
        invariant="Lorentz geodesic interp/rollout keeps L(x,x)=-1, upper sheet",
    ),
    dict(
        name="mclip_learnable_curvature",
        repo="wubumath",
        cmd="bin/test_manifold_clip",
        invariant="curvature moves during training and stays in sane band",
    ),
    dict(
        name="beam_mask_and_invisible_reader",
        repo="wubumath",
        cmd="bin/test_beam",
        invariant="visibility mask matches registry; invisible reader recovers all cells",
    ),
    dict(
        name="perceptual_band_localization",
        repo="wubumath",
        cmd="bin/test_bands",
        invariant="bass energy in band0, treble in band3, normalize preserves texture order",
    ),
    dict(
        name="quat_exp_log_round_trip",
        repo="wubumath",
        cmd="bin/test_quat_prop",
        invariant="log(exp(v))==v; slerp monotone; unit*product stays unit (fuzzed)",
    ),
    dict(
        name="kodak_audio_image_round_trip",
        repo="wubumath",
        cmd="bin/test_kodak",
        invariant="audio->image->audio corr>0.99 through 256x256 RGB layout",
    ),
    dict(
        name="pair_corpus_val_generalization",
        repo="wubumath",
        cmd="bin/test_pairs",
        invariant="recall@1/@5 on held-out val scenes above chance, monotone in k",
    ),
    dict(
        name="pframe_residual_rd_monotone",
        repo="wubumath",
        cmd="bin/test_flow_matching",
        invariant="residual recon error decreases with quant levels; exact bit count",
    ),
    dict(
        name="pframe_infrared_pipeline",
        repo="wubumath",
        cmd="bin/test_pframe_ir",
        invariant="P-frame residual survives beam IR round trip; renderer blind; RD monotone",
    ),
    dict(
        name="hyperbolic_attention_primitives",
        repo="wubumath",
        cmd="bin/test_hattn",
        invariant="softmax matching sums to 1; gyromidpoint on-ball; identical-points identity",
    ),
    dict(
        name="lorentz_clip_hyperboloid",
        repo="wubumath",
        cmd="bin/test_lorclip",
        invariant="Lorentz lift keeps L=-1 upper sheet; distance symmetric; contrastive beats chance",
    ),
    dict(
        name="mclip_meru_stability_recipe",
        repo="wubumath",
        cmd="bin/test_manifold_clip",
        invariant="entail_weight honored; per-modality alphas learn; tau floor + curvature clamp held",
    ),
    dict(
        name="uv_band_provenance",
        repo="wubumath",
        cmd="bin/test_uv",
        invariant="UV watermark lossless; three bands coexist without cross-contamination",
    ),
    dict(
        name="stft_round_trip_reversible",
        repo="wubumath",
        cmd="bin/test_stft",
        invariant="ISTFT(STFT(x)) ~= x at hop=N/4 (corr>0.999)",
    ),
]

def run_seed(prop, seed):
    env = dict(os.environ)
    env["PROP_SEED"] = str(seed)          # suites read this when supported
    t0 = time.time()
    r = subprocess.run(prop["cmd"], shell=True, cwd=WUBUMATH,
                       env=env, capture_output=True, text=True)
    ok = (r.returncode == 0)
    return dict(seed=seed, ok=ok,
                rc=r.returncode,
                tail=(r.stdout[-400:] if not ok else ""),
                secs=round(time.time()-t0, 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = []
    all_ok = True
    for prop in PROPERTIES:
        runs = [run_seed(prop, s) for s in range(args.seeds)]
        ok = all(r["ok"] for r in runs)
        all_ok &= ok
        results.append(dict(property=prop["name"],
                            invariant=prop["invariant"],
                            repo=prop["repo"],
                            seeds=args.seeds,
                            status="HOLD" if ok else "VIOLATED",
                            runs=runs))
        print(f"[{'HOLD ' if ok else 'VIOLATED'}] {prop['name']:44s} "
              f"({args.seeds} seeds)  {prop['invariant'][:60]}")

    cert = dict(when=int(time.time()),
                seeds_per_property=args.seeds,
                total_properties=len(PROPERTIES),
                violations=sum(1 for r in results if r["status"]!="HOLD"),
                status="ALL_HOLD" if all_ok else "VIOLATIONS_PRESENT")
    print(f"\nProperty gate: {cert['status']} "
          f"({cert['total_properties']} properties x {args.seeds} seeds)")

    if args.json:
        with open(args.json,"w") as f:
            json.dump(dict(certificate=cert, results=results), f, indent=1)
        print(f"certificate -> {args.json}")

    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
