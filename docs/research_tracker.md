# Research Progress Tracker

**Last updated:** 2026-10-03

## Current Phase

**C1 RGB-only model verification, followed by C2 RGB+audio, C3 context definition, and a matched C1/C2/C3 retention comparison.**

RGB training features are prepared, but the supplied C1 training and result
summary is not supported by local checkpoints, logs, or result files. The local
Ryzen AI 7 350 / Radeon 860M was not detected by the installed ROCm 7.2.1
PyTorch build. Use a supported accelerator or retrieve and validate the
external run artifacts before marking C1 complete.

## Completed Work

| Item | Status |
|---|---|
| Environment, repository, and checkpoint audits | Complete |
| Official test feature acquisition and RGB/audio alignment | Complete |
| Pretrained XDVioDet MIX2 audio-video inference on 800 feature-listed videos | Complete |
| Video-level score calibration on validation data | Complete |
| Temporal scores and official-annotation mapping for the 218-video intersection | Complete |
| Official-interval temporal policy optimization | Complete, PILOT |
| C0/C2/C6 official-event baseline comparison, including 100 C6 seeds | Complete, PILOT |
| Raw-video FFmpeg pipeline validation on a selected 24-video subset | Complete, PILOT |
| RGB I3D training-feature preparation (19,770 files) | Complete; data preparation only |
| C1 RGB-only model training, evaluation, scoring, and retention | Pending artifact verification/reproduction |
| Newly trained C2 RGB+audio baseline | Not started |
| C3 context schema and context-conditioned VAS | Not started |

## Verified Score Results

The 800-video feature-listed inference partition has 500 positive and 300
negative weak video labels. These are project-reproduced scores, not a newly
trained detector and not a claim that the project split is the official
benchmark evaluation protocol.

| Metric | Value |
|---|---:|
| Offline AP / ROC-AUC | 0.995995 / 0.992793 |
| Online AP / ROC-AUC | 0.992007 / 0.987093 |
| Video-level validation Brier, raw / calibrated | 0.263724 / 0.111801 |
| Video-level evaluation Brier, raw / calibrated | 0.227950 / 0.102683 |

Video-level calibration used 160 validation and 640 evaluation videos. The
fixed detector is MIX2 audio-video: 1,024-D I3D RGB plus 128-D VGGish audio.

## Temporal Pilot Scope

The 218-video temporal set is the intersection of the 800 IDs in the test RGB
list with IDs in the official annotation file. The temporal files contain 44
validation videos and 174 evaluation videos, with 20,648 score units overall.
The 218 are not the complete 800-video set and are not described here as a
representative sample. The raw-video proof set is a separate, deliberately
selected subset of 24 videos.

The corrected event definition uses each official annotation frame-interval
pair as one event. An event is preserved when retained footage covers at least
80% of its interval. The selected temporal pilot configuration is threshold
0.12, a three-unit smoothing window, a two-unit minimum run, and zero pre/post
buffers. It corresponds to the 0.97 validation-event-recall target row despite
the historical `official_event_recall_95_policy` filename. All results below
remain **PILOT**.

### C0/C2/C6 Evaluation Pilot

Scope: the held-out 174-video evaluation split, 16,480 temporal units, and 357
official annotation intervals. C2 is conditioned on calibrated MIX2
audio-video scores. C6 samples the same total temporal-unit count as C2 for each
of 100 seeds, with seeds 20261002 through 20261101.

| Condition | Event recall | Event-duration coverage | Planned duration reduction |
|---|---:|---:|---:|
| C0 continuous retention | 0.991597 | 0.999229 | 0.00% |
| C2 calibrated MIX2 score-conditioned retention | 0.955182 | 0.963249 | 24.84% |
| C6 random duration-matched, 100 seeds | 0.380700 ± 0.022490 | 0.750330 ± 0.004164 | 24.84% |

C0 retains all 16,480 available evaluation units; three official intervals do
not meet the 80% coverage criterion. The C6 matching is by aggregate temporal
unit count (each unit has the same configured duration), not per-video
duration.

### Corrected Official-Interval Operating Points (PILOT)

| Validation target | Selected threshold | Smoothing / minimum run | Pre/post buffers | Evaluation event recall | Event-duration coverage | Retained units | Planned reduction | Missed / total events |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.16 | 1 / 2 | 0 / 0 | 0.871148 | 0.902598 | 61.35% | 38.65% | 46 / 357 |
| 0.95 | 0.13 | 3 / 2 | 0 / 2 | 0.941176 | 0.954647 | 71.72% | 28.28% | 21 / 357 |
| 0.97 | 0.12 | 3 / 2 | 0 / 0 | 0.955182 | 0.963249 | 75.16% | 24.84% | 16 / 357 |
| 0.98 | 0.11 | 1 / 1 | 0 / 0 | 0.991597 | 0.999229 | 100.00% | 0.00% | 3 / 357 |

## Raw-Video Pipeline Pilot

The selected 24-video proof subset produced 27 clips. The final FFmpeg output
was video-only because audio-inclusive AAC conversion failed on source audio
with a six-channel layout. This describes storage encoding only; the retention
decisions still used MIX2 audio-video scores.

| Measure | Result |
|---|---:|
| Duration retained / planned duration reduction | 9.588% / 90.412% |
| Normalized video-only byte reduction | 85.455% |
| Official events preserved in raw subset | 11 / 20 |
| Raw-subset event recall / event-duration coverage | 0.550000 / 0.680398 |

This is pipeline validation on a selected subset, not full-dataset performance.

## C1/C2/C3 Comparison Flow

1. **C1 RGB-only:** The 19,770 RGB training features and local list are
	prepared. Model training and the supplied checkpoint/PR-AUC/retention
	results remain unverified because the named checkpoint and result files are
	absent from the workspace. The reported NVIDIA T4 run requires its run log
	and outputs before it can be recorded as a measurement.
2. **C2 RGB+audio:** Train a distinct MIX2 model using 1,024-D RGB and 128-D
	VGGish audio features. Existing pretrained MIX2 score and retention pilots
	are not a substitute for a newly trained C2 baseline.
3. **C3 RGB+audio+context:** Define the context variables and their assignment
	source before implementing or evaluating context-conditioned VAS.
4. **Matched comparison:** Freeze the dataset split, epoch count, batch size,
	evaluation metrics, and retention budgets across C1/C2/C3 before comparing
	outcomes. No final three-configuration comparison exists yet.

## Important Limitations

- C1 RGB/video-only features are prepared; model training and supplied results remain unverified locally.
- C2 is calibrated MIX2 audio-video-score-conditioned retention, not video-only retention.
- Context assignment and context-conditioned VAS are not complete.
- IPFS, Fabric, encryption, integrity, and tamper-verification experiments are not complete.
- The 218-video retention results and 24-video raw-video results are PILOT and must not be represented as full-800-video results.
- Raw clips omit audio; detector scores do not.
- Earlier contiguous-positive-unit event groupings are superseded by the official annotation-interval definition.

## Next Step

Verify or reproduce C1 RGB-only training and obtain its checkpoint/evaluation
artifacts; then train C2, define C3 context variables, and run a matched
C1/C2/C3 comparison. The reported C1 metrics remain excluded until verified.

## Resume Prompt

```text
Continue by verifying or reproducing C1 RGB-only model results, then train C2
RGB+audio, define context for C3, and run a matched comparison. RGB feature
preparation is complete, but the supplied C1 checkpoint and metrics are absent
from the workspace. C0/C2/C6 and raw-video results remain PILOT. Existing
retention pilots use MIX2 audio-video scores. The temporal set contains 218
annotation-matched videos from the 800-ID feature list; raw FFmpeg validation
uses a selected 24-video subset. Do not claim full-dataset retention or
completed context/integrity work.
```
