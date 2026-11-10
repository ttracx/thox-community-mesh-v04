# THOX Community Mesh, specification v0.4

Craig Ross and Tommy Xaypanya. No synthetic author.

Open [index.html](index.html). It is one file: the research specification and a virtual laboratory that runs the same admission planner. No install. No server. No network call.

This is a preprint. It is not peer reviewed, and it is not a production authorization. It states no ship date, no dollar price, no tokens per second, and no community-mesh service level.

## What you can do in the file

- Read the specification. Every placement figure was produced by the planner embedded in the file.
- Open the laboratory. Change context. Build a pool of hosts. D23, the code of record, is on the left. The inequality printed in the household report is on the right. They disagree on purpose.
- Recompute the metadata hash chain, or flip one committed field and watch the hash fail. The token loop does not wait on that chain.
- Download the chain JSON and the admission evidence.

Try one 64 GB host at context 32768. D23 leaves layers unplaced. The printed rule still says whole-request.

## Source

`src/engine.mjs` is the admission planner. `src/ledger.mjs` is the metadata ledger. The same text is embedded in `index.html`, so the file does not depend on this directory at read time.

Geometry fingerprint of the frozen Qwen3.6-27B-Q4_0 preset:

`b4302c31c5d0b72435044638ad961409d7d4feeb354d3dadad4fd2777bf955e6`

LaTeX companion: [paper/community-mesh-v04.tex](paper/community-mesh-v04.tex). Reading the specification does not require TeX.

## Labels

- MEASURED: only sentences quoted from the household MeshStack report v1.1.0. Not re-run here.
- CALCULATED: admission, credits, and the latency identity.
- SIMULATED: seeded churn, the grant drill, and the sixteen-block ledger.

An arXiv identifier does not exist. Do not index this on Hugging Face until one does.

Code of record for admission remains `ttracx/thox-meshstack-inference`, decisions D22 and D23. This repository does not amend them.

Your AI. Your Data. Your Rules.
