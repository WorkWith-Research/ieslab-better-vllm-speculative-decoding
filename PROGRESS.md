# Progress Log — [iESLAB] Better vLLM Speculative Decoding

GitHub Project: https://github.com/orgs/WorkWith-Research/projects/1 (projectV2 id `PVT_kwDOEyEOMs4Bl95-`)

## Project items (for API updates)

| Item ID | Issue | Title | Status |
|---|---|---|---|
| `PVTI_lADOEyEOMs4Bl95-zg-9nck` | #1 | Phase 0: Environment setup | In progress |
| `PVTI_lADOEyEOMs4Bl95-zg-9nc4` | #2 | Phase 1: Observation — fixed-K SD under load | Backlog |
| `PVTI_lADOEyEOMs4Bl95-zg-9neU` | #3 | Phase 2: Analysis — oracle gap for dynamic K | Backlog |
| `PVTI_lADOEyEOMs4Bl95-zg-9nfY` | #4 | Phase 3: Prototype — lightweight decision model | Backlog |

Status option IDs: Backlog=`f75ad846` Ready=`61e4505c` In progress=`47fc9ee4`
In review=`df73e18b` Done=`98236657`. Update via `scripts/set_item_status.sh <item-id> <option-id>`.

## Log

### 2026-10-07
- Repo initialized; literature scan complete → `docs/literature-map.md`
  (key prior work: Nightjar, DSDE, DSpark scheduler, TETRIS, BanditSpec; gap =
  joint state-aware decisions + prefill/KV coupling).
- Observation plan drafted → `docs/observation-plan.md` (W1–W3 workloads, metric set,
  analyses Q1–Q4, deliverables).
- Phase issues #1–#4 created and added to project; #1 In progress.
- Env: vLLM 0.31.0 in `.venv` (py3.12); fixed torch cu130→cu129 for driver compat.
- Downloading Qwen2.5-7B-Instruct + leptonai EAGLE draft.
