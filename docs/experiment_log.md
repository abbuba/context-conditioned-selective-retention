# Experiment Log

Create one section for every executed experiment.

## Template

| Field | Value |
|---|---|
| Date | |
| Experiment ID | |
| Dataset/version | |
| Dataset split | |
| Git commit | |
| Random seed | |
| Hardware | |
| Video model | |
| Audio model | |
| Fusion method | |
| Calibration method | |
| Context-policy version | |
| Threshold-selection method | |
| Runtime | |
| Output directory | |
| Known failures | |
| Interpretation | |

## Dataset Preparation: XD-Violence Feature Directories

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | DATA-PREP-001 |
| Dataset | XD-Violence |
| Action | Created local directories for official test feature acquisition |
| Required modality | I3D RGB visual features and VGGish audio features |
| Checkpoint mode | MIX2 |
| Optical-flow features required | No |
| Training planned | No |
| Output directories | `data/raw/xd_violence/official_features/i3d_rgb_test`, `data/raw/xd_violence/official_features/vggish_audio_test` |
| Status | Complete |


## Feature Acquisition and Alignment: XD-Violence

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | DATA-VALIDATION-001 |
| Dataset | XD-Violence test partition |
| I3D RGB files | 4,000 |
| VGGish audio files | 800 |
| Matched videos | 800 |
| RGB crops per video | 5 |
| Optical flow used | No |
| Malformed RGB groups | 0 |
| Annotation lines | 500 |
| Free disk after extraction | 119.07 GB |
| Validation status | PASS |
| Output | `data/metadata/xd_violence_test_feature_manifest.csv` |

## Test Feature Lists

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | DATA-VALIDATION-002 |
| Matched videos | 800 |
| RGB list entries | 4,000 |
| Audio list entries | 800 |
| RGB-to-audio mapping | One audio feature per five RGB crops |
| Modality | MIX2 |
| Combined feature dimension | 1,152 |
| Status | PASS |
| Output | `external/XDVioDet/list/rgb_test.list`, `external/XDVioDet/list/audio_test.list` |
## Phase 2 Completion: Fixed Audio-Video Score Generation

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Status | Complete |
| Baseline | XDVioDet MIX2 pretrained checkpoint |
| Modality | I3D RGB + VGGish audio |
| Videos scored | 800 |
| Raw score output | `outputs/scores/xdviodel_mix2_raw_video_scores.csv` |
| Calibrated score output | `outputs/scores/xdviodel_offline_calibrated_event_probabilities.csv` |
| Offline AP | 0.995995 |
| Offline ROC-AUC | 0.992793 |
| Validation Brier raw → calibrated | 0.263724 → 0.111801 |
| Evaluation Brier raw → calibrated | 0.227950 → 0.102683 |
| Context used | No |
| Notes | Current score is audio-video MIX2, not video-only |

## Official-Interval Temporal Policy Optimization (PILOT)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | RETENTION-TEMPORAL-OFFICIAL-001 |
| Dataset/scope | XD-Violence annotation-matched temporal subset; 218 videos total |
| Split | 44 validation videos; 174 evaluation videos |
| Temporal units | 4,168 validation; 16,480 evaluation |
| Event definition | Official annotation frame-interval pairs |
| Event preservation | At least 0.80 temporal overlap with each official interval |
| Score modality | Calibrated XDVioDet MIX2 audio-video scores |
| Selection | Minimize validation retained-unit ratio among policies meeting each validation event-recall target |
| Status | PASS, PILOT only |
| Outputs | `outputs/metrics/xd_official_event_policy_operating_points.csv`, `outputs/logs/xd_official_event_policy_optimization.json` |
| Scope note | The 218 are annotation-ID matches from the 800-ID test-feature list; not all 800 and not a representative sample. |

| Validation event-recall target | Threshold | Smoothing / min-run | Pre/post buffer | Evaluation event recall | Event-duration coverage | Retained units | Planned reduction | Missed / total evaluation events |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.16 | 1 / 2 | 0 / 0 | 0.871148 | 0.902598 | 61.35% | 38.65% | 46 / 357 |
| 0.95 | 0.13 | 3 / 2 | 0 / 2 | 0.941176 | 0.954647 | 71.72% | 28.28% | 21 / 357 |
| 0.97 | 0.12 | 3 / 2 | 0 / 0 | 0.955182 | 0.963249 | 75.16% | 24.84% | 16 / 357 |
| 0.98 | 0.11 | 1 / 1 | 0 / 0 | 0.991597 | 0.999229 | 100.00% | 0.00% | 3 / 357 |

The frozen configuration used by later raw-subset processing is the 0.97
validation-target row (threshold 0.12, smoothing 3, minimum run 2, no buffers).
The policy JSON name `official_event_recall_95_policy` is inconsistent with
that operating-point row; do not use the filename as the policy target label.

## C0/C2/C6 Official-Event Baselines (100-Seed PILOT)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | RETENTION-C0-C2-C6-100-PILOT |
| Dataset/scope | Evaluation split of the 218-video temporal experiment |
| Evaluation videos / temporal units | 174 / 16,480 |
| Official evaluation events | 357 |
| Event-preservation threshold | 0.80 interval coverage |
| C2 input | Frozen calibrated MIX2 audio-video score policy |
| C6 seeds | 100; seed range 20261002–20261101 |
| Status | PASS, PILOT only |
| Outputs | `outputs/metrics/xd_c0_c2_c6_baseline_summary.csv`, `outputs/metrics/xd_c6_random_duration_matched_runs.csv`, `outputs/logs/xd_c0_c2_c6_baseline_report.json` |

| Condition | Event recall | Event-duration coverage | Planned duration reduction |
|---|---:|---:|---:|
| C0 continuous retention | 0.991597 | 0.999229 | 0.00% |
| C2 calibrated MIX2 score-conditioned retention | 0.955182 | 0.963249 | 24.84% |
| C6 random duration-matched, mean ± sample SD | 0.380700 ± 0.022490 | 0.750330 ± 0.004164 | 24.84% |

C0 retains all available evaluation temporal units; three official events do not
meet the 80% interval-coverage rule. C6 samples the same total number of
temporal units as C2 for every seed. Since units have a common duration this
matches aggregate duration, not retained duration per video.

## Frozen-Policy Application to Temporal Scores (PILOT)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | RETENTION-FROZEN-POLICY-218-PILOT |
| Scope | 218 annotation-matched videos; 20,648 temporal units |
| Retained units / aggregate retained-unit ratio | 15,715 / 76.11% |
| Planned aggregate duration reduction | 23.89% |
| Retention intervals | 356 |
| Status | PASS, PILOT only |
| Outputs | `outputs/metrics/xd_final_official_event_policy.json`, `outputs/metrics/xd_final_retention_decisions_all_218.csv`, `outputs/metrics/xd_final_retention_intervals_all_218.csv`, `outputs/metrics/xd_final_retention_split_summary.csv`, `outputs/logs/xd_final_policy_application.json` |
| Scope note | Do not describe this 218-video artifact as the full 800-video test-feature partition. |

## Raw-Video FFmpeg Pipeline Validation (PILOT)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | RAW-VIDEO-FINAL-POLICY-24-PILOT |
| Scope | Deliberately selected raw-video pipeline subset, 24 videos; not representative |
| Frozen score policy | MIX2 audio-video scores; threshold 0.12, smoothing 3, minimum run 2, zero buffers |
| Encoded outputs | 27 retained clips; video-only output (audio, metadata, chapters excluded) |
| Raw duration / retained duration | 2,141.518 s / 205.333 s |
| Duration retained / planned duration reduction | 9.588% / 90.412% |
| Normalized video-only baseline / retained clip bytes | 102,191,845 / 14,863,766 |
| Normalized byte reduction | 85.455% |
| Official events preserved in subset | 11 / 20; event recall 0.550000; event-duration coverage 0.680398 |
| Status | PASS for pipeline execution, PILOT for measured outcomes |
| Outputs | `outputs/logs/xd_final_policy_raw_clip_extraction.json`, `outputs/logs/xd_final_policy_raw_evidence_recall.json`, `outputs/raw_video_pipeline/xd_final_policy_retained_clip_manifest_24.csv` |
| Limitation | Audio-inclusive AAC conversion failed on source six-channel audio; stored clips are video-only. Detector scores remain MIX2 audio-video. |

## Pending Work

- C1 RGB-only baseline; successful AMD ROCm training has not been verified.
- Context-schema definition and context-conditioned VAS evaluation.
- IPFS, Fabric, encryption, integrity, and tamper-verification experiments.
- Documentation of the 218-video annotation intersection and 800-video feature scope in final study materials before any retention result is described as final.

## C1 RGB Training Feature Preparation and Result Verification

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Experiment ID | DATA-PREP-RGB-TRAIN-001 |
| Dataset/modality | XD-Violence I3D RGB training features |
| Source | `data/raw/xd_violence/official_features/archives/i3d-features.zip`, members under `i3d-features/RGB/` |
| Original list entries | 19,770 in `external/XDVioDet/list/rgb.list` |
| Prepared files | 19,770; 20,931,390,720 extracted bytes |
| Local list | `external/XDVioDet/list/rgb_train_local.list` |
| Validation | PASS; 0 missing list features, 0 zero-byte files, 0 size mismatches, 0 duplicate basenames |
| Preparation report | `outputs/metrics/xd_rgb_train_feature_preparation.json` |
| Interpretation | Training-feature preparation only; it does not establish that an RGB model was trained or evaluated. |

### C1 Training/Evaluation Claim Verification (2026-10-03)

The supplied C1 summary claims training on an NVIDIA Tesla T4 and reports a
`c1_rgb.pkl` checkpoint, checkpoint-evaluation and score CSVs, and retention
results. These claims are not entered as experimental results because the
corresponding checkpoint, run log, evaluation CSVs, score files, and retention
table are absent from the inspected project artifacts. The local
`external/XDVioDet/ckpt` contains only `wsanodet_mix2.pkl`; the named C1 files
were not found in the project outputs. C1 model training, checkpoint selection,
scoring, and retention therefore remain unverified in this workspace.
