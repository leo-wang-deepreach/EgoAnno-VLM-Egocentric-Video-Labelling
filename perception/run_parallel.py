#!/usr/bin/env python3
"""run_parallel.py — shard grasp-time frames across N ground_v16 worker processes (each loads its own
SAM model) and merge the results. Per-frame grounding is independent, so this ~Nx the wall-clock
(GPU-memory permitting; falls back if a worker dies). Workers inherit env (DEPTH_DIR etc.).

Run: ../.venv/bin/python perception/run_parallel.py <outdir> <inventory.json> <prompts.json> [N=3]
"""
import json
import os
import shutil
import subprocess
import sys
import time

SAM3PY = "/home/ubuntu/local/sam3/sam3py"
HERE = os.path.dirname(os.path.abspath(__file__))
STAGGER = float(os.environ.get("STAGGER", "18"))    # seconds between worker launches (avoid simultaneous model-load OOM)


def _log(m):
    print(m, flush=True)


def main():
    outdir, inv, prompts_file = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    os.makedirs(outdir, exist_ok=True)
    pj = json.load(open(prompts_file))
    times = sorted({round(p["t"], 3) for p in pj["prompts"]})
    shards = [set(times[i::n]) for i in range(n)]              # round-robin times across workers
    procs = []
    launched = [s for s in shards if s]
    for k, sh in enumerate(shards):
        sub = {kk: pj[kk] for kk in ("video", "W", "H", "fps") if kk in pj}
        sub["prompts"] = [p for p in pj["prompts"] if round(p["t"], 3) in sh]
        if not sub["prompts"]:
            continue
        tf = os.path.join(outdir, f"_prompts_shard{k}.json")
        json.dump(sub, open(tf, "w"))
        sd = os.path.join(outdir, f"shard{k}"); os.makedirs(sd, exist_ok=True)
        log = open(os.path.join(outdir, f"_w{k}.log"), "w")
        p = subprocess.Popen([SAM3PY, os.path.join(HERE, "ground_v16.py"), sd, inv, tf],
                             stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        procs.append((k, p, sd))
        _log(f"worker {k}: {len(sub['prompts'])} prompts ({len(sh)} times) pid={p.pid}")
        if len(procs) < len(launched):                         # STAGGER: let this model finish loading first
            time.sleep(STAGGER)
    bad = 0
    for k, p, sd in procs:
        rc = p.wait()
        _log(f"worker {k} exited rc={rc}")
        if rc != 0:
            bad += 1
    if bad:
        _log(f"WARNING: {bad}/{len(procs)} workers exited nonzero — merged frames may be incomplete")
    # merge PNGs + index shards
    index = []; tag = json.load(open(inv))["tag"]
    for k, p, sd in procs:
        if not os.path.isdir(sd):
            continue
        for fn in os.listdir(sd):
            if fn.endswith(".png"):
                shutil.move(os.path.join(sd, fn), os.path.join(outdir, fn))
            elif fn.startswith("_index_"):
                index += json.load(open(os.path.join(sd, fn)))
    json.dump(index, open(os.path.join(outdir, f"_index_{tag}.json"), "w"), indent=2)
    g = sum(1 for r in index if r["obj"])
    print(f"\nmerged {len(index)} hand-frames from {len(procs)} workers | grounded {g} | N/A {len(index) - g}")


if __name__ == "__main__":
    main()
