# Current Verified Methodology Draft — 2026-10-03

The material in this section is the current evidence-grounded methodology
draft. Older duplicated drafts below are retained as history but are
superseded; do not use their prospective context, dataset, buffer, encryption,
or storage statements as completed-study claims.

## III. Methodology

### A. System Scope and Research Stages

The completed score pipeline uses a fixed pretrained XDVioDet MIX2 audio-video
checkpoint to produce offline and online event scores. The project calibrated
offline scores, extracted temporal scores, and evaluated context-free retention
policies as pilots. A separate FFmpeg pipeline was validated on a selected
24-video raw-video subset. These stages do not constitute a completed
context-conditioned Video Anomaly Score (VAS) study.

The checkpoint was not retrained, and the model architecture, weights, input
features, and score formulation were not changed. A local CPU compatibility
patch changed device placement only. RGB training features are now prepared,
but no C1 checkpoint, training log, evaluation CSV, score output, or retention
table is present in the inspected project artifacts. The C1 model results
supplied separately therefore remain unverified and are not reported here.
IPFS, Hyperledger Fabric, encryption, integrity, and tamper-verification
experiments are not complete.

### B. Dataset and Scope

The project has 800 matched XD-Violence test-feature video identities, with
4,000 RGB feature-list entries and 800 audio feature-list entries. Project weak
video labels comprise 500 anomalous and 300 normal videos. The project-created
160/640 calibration/evaluation split is not the official benchmark evaluation
protocol.

The temporal-retention artifacts contain 218 videos: the intersection of the
800 test-feature IDs and IDs in the official annotation file. Those artifacts
split into 44 validation and 174 evaluation videos, with 20,648 temporal score
units. The 218-video set is not the full 800-video set and is not described as a
representative sample. The raw-video pipeline instead uses a deliberately
selected 24-video proof subset; it is a stress/pipeline validation subset, not
a benchmark performance sample. XD-Violence movie/web-video clips are not
described as deployed CCTV footage.

### C. Fixed Pretrained Audio-Video Detector

The event-score generator is the official XDVioDet `wsanodet_mix2.pkl`
checkpoint, loaded and audited against the official repository implementation.
The checkpoint SHA-256 recorded by the project is
`27067896258294da0a711e3f0c071190ea5848961298cbde389265627191ec2c`. It is a
fixed pretrained model, not a newly trained detector. Offline and online scores
in this work are reproduced project inference results and are not presented as
a new official benchmark submission.

### D. RGB and Audio Features

MIX2 concatenates 1,024-dimensional I3D RGB features and 128-dimensional
VGGish audio features, yielding 1,152 input dimensions. The test feature lists
contain five RGB crops per video and one corresponding audio sequence; the
XDVioDet loader maps the audio sequence to those five RGB crops. Optical-flow
features are not used in MIX2. For the planned C1 RGB-only experiment, the
project has prepared 19,770 I3D RGB training feature files and a Windows-local
training list; the preparation manifest reports no missing files. This is data
preparation, not evidence of model training. No C1 checkpoint, training log,
checkpoint-evaluation CSV, video- or segment-score CSV, or retention-budget
table was found in the project artifacts during the 2026-10-03 documentation
audit. Consequently, C1 model performance and the supplied C1 retention values
remain unverified and are not reported as results here.

#### C1–C3 Experimental Sequence

C1 is the RGB-only XDVioDet configuration (1,024-D I3D RGB input). Its local
training feature preparation is verified, while the supplied training,
checkpoint, evaluation, scoring, and retention claims await their artifacts.
C2 is planned as a separately trained RGB+audio MIX2 model (1,024-D RGB plus
128-D VGGish audio). The earlier fixed-pretrained MIX2 inference and retention
pilots are not results for this newly trained C2 configuration. C3 adds
operational context only after the context variables, assignment provenance,
and policy are defined; these are not yet available.

For a valid C1/C2/C3 comparison, the split, epoch count, batch size, evaluation
metrics, and retention budgets must be fixed consistently across all three
configurations before reporting comparative results. No such completed matched
comparison is currently documented.

### E. Temporal Score Mapping

For each temporal index, the model applies sigmoid to the five crop outputs and
averages across crops, retaining one offline and one online score per temporal
unit. The official evaluator repeats predictions 16 times; consequently,
temporal index \(k\) maps to frame interval
\([16k,16(k+1)-1]\). The project uses 24 FPS for estimated seconds, or
\(16/24\) seconds per unit. This frame mapping is supported by the evaluator;
the seconds conversion is an estimate and is not claimed as exact encoded-video
timing for all 218 videos.

Official anomaly events are the frame-interval pairs in the annotation file.
Inclusive end frame \(e\) is represented as the half-open time interval
\([s/24,(e+1)/24)\). The corrected retention optimizer evaluates coverage of
each official interval; earlier contiguous-positive-unit event grouping is
superseded.

### F. Score Calibration and Split Discipline

Video-level Platt scaling is implemented as logistic regression on offline raw
scores. It is fit using only the 160-video validation split and applied without
refitting to the 640-video evaluation split. Temporal calibration is a
separate logistic regression fit on temporal units from the 44-video
validation subset and applied to the 174-video evaluation subset. These two
calibrators operate at different granularities and their metrics must not be
combined. Threshold and policy selection for temporal retention uses the
validation subset; the selected policy is then evaluated on the held-out
evaluation subset.

### G. PILOT Retention Policies

For official event \(G_j\) and retained interval union \(\mathcal{P}\), event
coverage is

\[
C_j=\frac{|G_j\cap\mathcal{P}|}{|G_j|}.
\]

An event is counted as preserved when \(C_j\geq0.80\). Official-event recall
is preserved official intervals divided by all official intervals. Event-
duration coverage is the total covered duration of official intervals divided
by their total duration.

The corrected operating-point search evaluates thresholds with temporal score
smoothing, minimum trigger-run length, and pre/post buffers. It selects the
lowest-retention validation candidate meeting each target validation
official-event recall, then evaluates that candidate on held-out data. The
frozen policy used in subsequent pilot processing is the 0.97-target row:
threshold 0.12, smoothing window 3 units, minimum run 2 units, and zero pre/post
buffers. Its observed validation recall is 0.976190. A generated policy JSON
uses a historical “95” name for this configuration; the operating-point CSV
identifies its actual target as 0.97.

C0 retains every available evaluation temporal unit. C2 applies the frozen
calibrated MIX2 audio-video score policy. C6 uniformly samples the same total
number of temporal units as C2 and repeats the random selection over 100 seeds
(20261002–20261101). Since temporal units have equal configured duration, C6
matches aggregate retained duration, not the per-video retained duration
distribution. C0/C2/C6 values remain PILOT results.

### H. Raw-Video Pipeline Measurement

The raw-video FFmpeg pipeline was evaluated on a selected 24-video subset. It
extracts policy intervals and compares retained clip bytes with a normalized
full-video baseline using the same video-only encoding scope. Audio, metadata,
and chapters are excluded from both encoded outputs. Audio-inclusive conversion
failed on source six-channel audio with the available AAC encoder; consequently,
the stored outputs omit audio for technical compatibility. This does not make
the detector video-only: the retention decisions use MIX2 audio-video scores.

The raw-video byte and event-coverage measurements describe only this selected
24-video pipeline proof subset. They are not full-dataset storage reduction or
final event-recall estimates. The subset was deliberately selected to include
retained, missed, near-threshold, and normal cases.

### I. Metrics and Claim Boundaries

The Brier score for probabilities \(p_i\) and binary labels \(y_i\) is
\(N^{-1}\sum_i(p_i-y_i)^2\). Average precision and ROC-AUC are computed from
the project’s specified scores and labels. Retained-unit percentage is
\(100\times U_r/U\); planned duration reduction is
\(100\times(1-U_r/U)\). Raw-pipeline normalized-byte reduction is
\(100\times(1-B_c/B_n)\), where \(B_c\) is retained video-only clip bytes and
\(B_n\) is normalized video-only full-video baseline bytes. The latter metric
is available only for the selected 24-video subset.

Operational context has not been assigned or evaluated. Therefore, these
experiments do not validate context-conditioned VAS. No claims of encryption,
immutable video, IPFS storage, Hyperledger Fabric anchoring, hashing-based
tamper detection, privacy guarantees, or integrity validation are supported by
the current artifacts.

## Archived Pre-Results Draft (Superseded)

The material below this heading is retained for project history. It includes
earlier proposed designs and repeated draft text; the verified scope and
methodology above supersede it. Do not interpret its proposed datasets, context
formulas, buffer settings, or planned systems work as executed experiments.

## Experimental Protocol and Reproducibility

The study was designed as a reproducible evaluation of selective evidence
preservation in surveillance video streams. Rather than evaluating only
event-detection accuracy, the experimental protocol evaluates whether a
preservation policy can reduce retained footage while maintaining coverage of
temporally annotated relevant events.

All experiments are organized into predefined conditions: continuous retention;
video-only event-triggered retention; audio-video event-triggered retention;
context-conditioned event scoring; context-conditioned preservation; rule-based
policy retention; duration-matched random retention; and periodic retention.
Model configurations, preservation thresholds, temporal-window parameters,
random seeds, and hardware settings are stored with each experimental run.

The primary evaluation dataset is XD-Violence because it provides long,
untrimmed surveillance-related videos with synchronized audio and video, which
supports multimodal event scoring and offline replay of retention decisions.
Scene-dependent policy experiments are additionally evaluated using NWPU Campus.
Temporal clip-boundary analysis is conducted using the CCTV subset of the NTU
CCTV-Fights dataset, which provides frame-level fight boundaries.

Dataset partitions are separated at the video level. All model selection, score
calibration, policy-weight selection, and preservation-threshold selection are
performed using validation data only. The final test partition is not used to
select any configuration.

## Selective Evidence Preservation Metrics

Let \(G_j=[g^{(j)}_{\mathrm{start}},g^{(j)}_{\mathrm{end}}]\) denote the
temporal interval of the \(j\)-th ground-truth relevant event, and let
\(\mathcal{P}=\{P_i\}\) denote the set of temporal intervals retained by a
preservation policy. The event-coverage ratio is defined as:

\[
\operatorname{Coverage}(G_j,\mathcal{P}) =
\frac{
\left|G_j \cap \left(\bigcup_i P_i\right)\right|
}{
|G_j|
}.
\]

An event is considered preserved when its coverage exceeds a pre-specified
threshold \(\eta\):

\[
\operatorname{Preserved}(G_j)=
\mathbf{1}\left[
\operatorname{Coverage}(G_j,\mathcal{P}) \geq \eta
\right].
\]

The evidence-recall metric is then:

\[
ER=
\frac{1}{N}
\sum_{j=1}^{N}
\operatorname{Preserved}(G_j),
\]

where \(N\) is the number of ground-truth relevant events.

To assess whether the retained clip includes lead-up and aftermath information,
each event is expanded using pre-event and post-event temporal margins:

\[
G_j^{+}
=
[
g^{(j)}_{\mathrm{start}}-\Delta_{\mathrm{pre}},
g^{(j)}_{\mathrm{end}}+\Delta_{\mathrm{post}}
].
\]

The retention ratio is:

\[
R_{\mathrm{ret}} =
\frac{
\left|\bigcup_iP_i\right|
}{
T_{\mathrm{total}}
},
\]

where \(T_{\mathrm{total}}\) is the total duration of the processed
surveillance stream. Storage reduction is calculated as:

\[
SR = 100 \times (1-R_{\mathrm{ret}}).
\]

The principal evaluation is the trade-off between evidence recall and retention
ratio. A desirable policy achieves high evidence recall while retaining a
smaller proportion of the original video than continuous recording and simpler
baseline policies.

## Computational Environment

The experimental framework was implemented in Python 3.12.10 on a 64-bit
Windows 11 workstation. The active Python interpreter was located at:

```text
C:\Users\abbub\OneDrive\Desktop\Experiment\myvenv\Scripts\python.exe
```

The local environment included PyTorch 2.14.1 configured for CPU execution,
OpenCV 5.0.0, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, PyYAML 6.0.3, and
FFmpeg 9.0.1. The workstation contained 31.29 GB RAM, with 17.2 GB available
at the time of the environment audit, and 167.8 GB free storage on the
experiment drive.

PyTorch did not detect a CUDA-compatible GPU in the local execution
environment. Accordingly, the local workstation was used for dataset auditing,
metadata construction, temporal retention simulation, metric computation,
evidence packaging, integrity verification, and controlled CPU-scale
experiments. GPU-dependent feature extraction and computationally intensive
model training were deferred to a separately documented accelerated
environment when required.

FFmpeg was available through the system path and was used to inspect video
streams, identify audio tracks, measure duration and frame rate, extract audio
when required, and construct temporally selected evidence clips. All hardware
properties, package versions, model configurations, random seeds, and
experiment outputs were recorded before execution to support reproducibility.

## Fixed Event-Score Baseline

A fixed pretrained XDVioDet model was used as the audio-visual event-score
generator. The selected official checkpoint was
`wsanodet_mix2.pkl`, obtained from commit
`f846d8cfc454943ec165bd2489b71e7dc0a1064d` of the official XDVioDet
repository. The SHA-256 checksum of the checkpoint was
`27067896258294da0a711e3f0c071190ea5848961298cbde389265627191ec2c`.

The model operates in MIX2 mode and receives concatenated I3D RGB visual
features of dimension 1024 and VGGish audio features of dimension 128,
resulting in a 1152-dimensional input representation. Optical-flow features
were not used. The detector was not retrained or modified; it was used only to
produce temporally indexed event scores for calibration and selective
evidence-preservation experiments.

## Feature Acquisition and Modality Alignment

For the XD-Violence test partition, 4,000 I3D RGB feature files and 800 VGGish
audio feature files were extracted from the official feature archives. The
visual representation uses five RGB crops per video, resulting in:

\[
N_{\mathrm{RGB}} = 5N_{\mathrm{video}}.
\]

The extracted files contained 800 matched video identities. Every matched video
had exactly five RGB crop files and one corresponding VGGish audio feature file.
The resulting alignment was therefore:

\[
N_{\mathrm{RGB}}=800\times5=4000,
\qquad
N_{\mathrm{audio}}=800.
\]

The XDVioDet MIX2 data loader associates the audio feature indexed by
\(\lfloor i/5\rfloor\) with the five RGB crops belonging to the same video.
The alignment validation found no missing audio files, unmatched video
identities, or incomplete five-crop groups. Optical-flow features were not used
in the present study.

The downloaded test annotations contained 500 non-empty annotation records.
These records were retained as the official temporal annotation source and were
not interpreted as a video-count field.

## Baseline Inference Configuration

The fixed XDVioDet checkpoint was evaluated using the official MIX2 input
configuration. The test feature lists contained 4,000 I3D RGB entries and 800
VGGish audio entries for 800 matched videos. Five RGB crop features were
associated with each video, while the corresponding VGGish feature was mapped
to the five visual crops.

The model input dimension was fixed at 1,152, consisting of 1,024 visual
features and 128 audio features. The checkpoint and model architecture were
not modified or retrained. The output of this inference stage consists of
temporally indexed event scores, which are subsequently calibrated before
being passed to the selective evidence-preservation policies.


## Inference Entrypoint

Inference was performed using the official `test.py` entrypoint from the
XDVioDet repository at commit `f846d8cfc454943ec165bd2489b71e7dc0a1064d`.
The checkpoint was evaluated in MIX2 mode using the verified I3D RGB and VGGish
feature lists. The exact command-line configuration was recorded before
execution, and the model weights and architecture were not modified.

The inference stage produces raw event scores before calibration. These raw
scores are retained as immutable experiment outputs and are used as inputs to
the subsequent score-calibration and selective-preservation experiments.

## Score Granularity Limitation

The official XDVioDet evaluation code produces a score for each feature input and
repeats the resulting score 16 times for comparison with the benchmark
ground-truth vector. Therefore, the official feature-level pipeline is used for
video-level event scoring, calibration, and retention-policy simulation.

These feature-level outputs are not treated as native frame-level predictions.
Temporal clip extraction, pre-event coverage, post-event coverage, encrypted
clip generation, and IPFS/Hyperledger Fabric evaluation will be conducted
separately on a documented raw-video subset. The manuscript distinguishes
feature-level policy simulation from raw-video evidence-lifecycle experiments.

## Fixed Score-Generation Protocol

The fixed pretrained XDVioDet MIX2 checkpoint was executed without retraining.
Inference was completed for 800 aligned test-partition videos using CPU
execution. The runner generated one offline and one online raw event score for
each video after averaging the sigmoid-transformed outputs across the five RGB
crop features associated with that video.

Because the study evaluates a preservation policy rather than detector training,
the locally available 800-video feature partition was divided using a fixed
stratified split. Twenty percent of videos were assigned to a validation subset
for score calibration and preservation-threshold selection, while the remaining
eighty percent were assigned to the policy-evaluation subset. The random seed
was fixed to 42. This split is used only for the proposed retention-policy
experiments and is not presented as the official XD-Violence benchmark
evaluation protocol.

## Score Calibration

Raw offline XDVioDet scores were calibrated using Platt scaling, implemented as
logistic regression fitted only on the 160-video validation subset. Let
\(S_{\mathrm{raw}}\) denote the raw offline score. The calibrated event
probability was defined as:

\[
p_e =
\sigma(\beta_0 + \beta_1 S_{\mathrm{raw}}),
\]

where \(\sigma(\cdot)\) is the logistic sigmoid function and
\(\beta_0,\beta_1\) were estimated using validation data only. The fitted
calibration model was subsequently applied without modification to the
640-video evaluation subset.

Calibration quality was measured using the Brier score, while discrimination
was reported using average precision and ROC-AUC. The calibrated event
probability is used as the event-evidence input to the subsequent
context-conditioned preservation-priority policy.

## Operational Context Representation

Operational context was defined as an externally specified and versioned policy
profile rather than as an inferred visual feature. The context vector is:

\[
\mathbf{c} =
[
\text{camera identifier},
\text{scene identifier},
\text{zone class},
\text{site type},
\text{operating state},
\text{time period},
\text{allowed activity},
\text{risk weight},
\text{policy version}
].
\]

This design separates observed event evidence from deployer-defined
operational-policy information. Site labels, operating schedules, and
restricted-zone policies are not inferred from videos when such metadata are
unavailable.

XD-Violence was used only for audio-video event-score calibration and
context-free retention-policy baselines because its feature files do not provide
verified operational-policy metadata. Context-conditioned preservation will be
evaluated separately using a dataset or controlled raw-video collection with
documented scene, camera, and policy information.

## Temporal Score Mapping Verification

The fixed pretrained XDVioDet MIX2 model was inspected before temporal
retention experiments. For a five-crop RGB batch, the model produced offline
and online outputs of shape \([5,T,1]\), where \(T\) is the number of temporal
feature units. The sigmoid-transformed outputs were averaged across the five
RGB crops, yielding one score per temporal feature index.

The official XDVioDet evaluation code repeats each temporal prediction 16
times when comparing predictions with ground-truth labels. Therefore, temporal
index \(k\) was mapped to the feature-frame interval
\([16k,16(k+1)-1]\). RGB I3D and VGGish audio feature arrays had identical
temporal lengths for every inspected video, enabling aligned audio-visual
score extraction.

A five-video proof experiment generated temporal scores and annotation-overlap
labels for 627 temporal units. Among the locally available 800 test feature
videos, 218 matched official temporal annotation records. The 16-frame mapping
is directly supported by the official evaluation implementation. Time in
seconds was estimated using the 24 FPS convention supported by the observed
feature-duration relationship; exact encoded-video timestamps require later
raw-video verification.

Below is a **methodology-only draft** for the completed score-generation, calibration, temporal-mapping, and pilot retention setup. It does not claim final results, VAS performance, context benefit, raw-video clip extraction, IPFS, Fabric, or storage-integrity outcomes.

Replace or adapt the corresponding methodology sections in your journal manuscript. Keep the headings suitable for an IEEE-style paper.

***

## III. Methodology

### A. Overview

This study investigates selective evidence preservation using a fixed pretrained audio-visual event detector as an event-score generator. The aim at this stage is not to introduce a new anomaly-detection architecture or retrain an existing detector. Instead, the system uses calibrated temporal audio-visual event evidence to support subsequent retention decisions.

The processing pipeline is:

\[
\text{I3D RGB Features}
+
\text{VGGish Audio Features}
\rightarrow
\text{Pretrained Audio-Visual Model}
\rightarrow
\text{Temporal Event Scores}
\rightarrow
\text{Score Calibration}
\rightarrow
\text{Retention Policy}
\]

The proposed long-term evidence pipeline is:

\[
\text{Temporal Event Scores}
\rightarrow
\text{Calibrated Event Probability}
\rightarrow
\text{Context-Conditioned Preservation Score}
\rightarrow
\text{Temporal Retention Interval}
\rightarrow
\text{Encrypted Evidence Clip}
\rightarrow
\text{SHA-256 Hash}
\rightarrow
\text{IPFS Storage}
\rightarrow
\text{Hyperledger Fabric Record}
\]

However, the present methodological stage focuses only on the first part of this pipeline: fixed pretrained event-score extraction, calibration, temporal score interpretation, and the preparation of selective-retention experiments. Context-conditioned VAS evaluation, raw-video clip extraction, IPFS storage, Hyperledger Fabric anchoring, and integrity verification are evaluated only after their respective inputs and policies are frozen.

### B. Dataset and Feature Representation

The audio-visual event-score generator is evaluated using the XD-Violence dataset, which was introduced for weakly supervised audio-visual violence detection. The dataset contains 4,754 untrimmed videos with synchronized audio signals and a total duration of approximately 217 hours. It includes normal and anomalous videos collected from multiple scenes and event categories. [dl.acm](https://dl.acm.org/doi/10.1007/978-3-030-58577-8_20)

The study uses official pre-extracted test-partition features rather than training a new visual or audio model. The selected pretrained baseline receives two aligned feature representations:

1. **Visual features:** I3D RGB features of dimension 1024.
2. **Audio features:** VGGish features of dimension 128.

For temporal index \(k\), the audio-visual feature vector is constructed as:

\[
\mathbf{x}_k =
[
\mathbf{x}^{\mathrm{RGB}}_k,
\mathbf{x}^{\mathrm{Audio}}_k
]
\]

where:

\[
\mathbf{x}^{\mathrm{RGB}}_k
\in
\mathbb{R}^{1024}
\]

and:

\[
\mathbf{x}^{\mathrm{Audio}}_k
\in
\mathbb{R}^{128}
\]

The fused feature dimension is therefore:

\[
d_{\mathrm{MIX2}}
=
1024 + 128
=
1152
\]

The official XD-Violence feature package contains five RGB crop features for every video and one temporally aligned VGGish audio feature sequence. For the extracted test partition:

\[
N_{\mathrm{video}}=800
\]

\[
N_{\mathrm{RGB\;feature\;files}}
=
5\times800
=
4000
\]

\[
N_{\mathrm{audio\;feature\;files}}
=
800
\]

Every video identity was verified to have exactly five RGB crops and one corresponding audio feature file. No missing audio files, unmatched identities, or incomplete RGB crop groups were observed during the feature-alignment audit.

### C. Fixed Pretrained Audio-Visual Baseline

A fixed pretrained XDVioDet model was used as the audio-visual event-score generator. The model corresponds to the official implementation associated with the XD-Violence benchmark. The selected checkpoint was:

```text
wsanodet_mix2.pkl
```

The implementation repository commit was:

```text
f846d8cfc454943ec165bd2489b71e7dc0a1064d
```

The checkpoint SHA-256 checksum was:

```text
27067896258294da0a711e3f0c071190ea5848961298cbde389265627191ec2c
```

The model operates in MIX2 mode, receiving concatenated I3D RGB and VGGish audio features. Optical-flow features were not used.

The baseline model was neither retrained nor architecturally modified. It was used exclusively to generate temporal audio-visual event scores. The model is therefore treated as an external fixed event detector, while the selective evidence-preservation policy constitutes the experimental focus.

The checkpoint input representation is:

\[
\mathbf{X}
=
[
\mathbf{x}_1,
\mathbf{x}_2,
\ldots,
\mathbf{x}_T
]
\in
\mathbb{R}^{T\times1152}
\]

where \(T\) is the temporal feature-sequence length for a given video.

### D. Device-Compatible Inference

The official XDVioDet implementation included hard-coded CUDA tensor placement inside its distance-adjacency component. The local execution environment used CPU-only PyTorch and therefore could not execute hard-coded CUDA operations.

To allow reproducible CPU inference, only the tensor-device placement logic was adapted. Hard-coded device calls of the form:

```python
.to('cuda')
```

were replaced with runtime-aware placement:

```python
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
```

and:

```python
.to(device)
```

This adjustment affected only the device on which distance-adjacency tensors were allocated. It did not modify:

- Model checkpoint weights.
- Model architecture.
- Parameter names.
- Input features.
- Feature dimensions.
- Temporal sequence length.
- Model score formulation.
- Training procedure.

The original implementation source file, patched source file, and SHA-256 hashes were retained in the experiment record.

### E. Temporal Audio-Visual Score Extraction

For every test video, the model receives five RGB crop feature sequences and one associated VGGish audio feature sequence. Let \(c\in\{1,\ldots,5\}\) denote the RGB crop index, and let \(k\in\{0,\ldots,T-1\}\) denote the temporal index.

The model produces offline and online logits with the form:

\[
\mathbf{L}^{\mathrm{offline}}
\in
\mathbb{R}^{5\times T\times1}
\]

\[
\mathbf{L}^{\mathrm{online}}
\in
\mathbb{R}^{5\times T\times1}
\]

The sigmoid transformation is applied to each logit:

\[
p_{c,k}^{\mathrm{offline}}
=
\sigma(
L_{c,k}^{\mathrm{offline}}
)
\]

\[
p_{c,k}^{\mathrm{online}}
=
\sigma(
L_{c,k}^{\mathrm{online}}
)
\]

where:

\[
\sigma(z)
=
\frac{1}{1+e^{-z}}
\]

The five RGB crop scores are then averaged while retaining the temporal dimension:

\[
S_k^{\mathrm{offline}}
=
\frac{1}{5}
\sum_{c=1}^{5}
p_{c,k}^{\mathrm{offline}}
\]

\[
S_k^{\mathrm{online}}
=
\frac{1}{5}
\sum_{c=1}^{5}
p_{c,k}^{\mathrm{online}}
\]

Thus, each video produces one offline and one online score for every temporal feature unit.

For video-level pilot analysis, temporal scores may be aggregated as:

\[
S_{\mathrm{video}}^{\mathrm{offline}}
=
\frac{1}{T}
\sum_{k=1}^{T}
S_k^{\mathrm{offline}}
\]

The video-level score is used only for early feasibility checks and score calibration. The temporal score sequence is retained for subsequent temporal-retention experiments.

### F. Temporal Feature-to-Frame Mapping

Temporal-score mapping was verified by inspecting official model outputs, original feature-array dimensions, the official evaluation procedure, and official temporal annotations.

The official XDVioDet evaluation code repeats each temporal prediction 16 times before comparing it with frame-level ground truth:

```python
np.repeat(pred, 16)
```

Accordingly, one temporal score corresponds to one 16-frame interval. For temporal index \(k\):

\[
f_{\mathrm{start}}(k)
=
16k
\]

\[
f_{\mathrm{end}}(k)
=
16(k+1)-1
\]

where \(f_{\mathrm{start}}(k)\) and \(f_{\mathrm{end}}(k)\) are the inclusive start and end frames represented by temporal score \(k\).

The observed relationship between feature-sequence length and clip duration supports the 24 FPS convention used for time estimation. For example, a 240-second clip produced 360 temporal feature units:

\[
360\times16
=
5760\text{ frames}
\]

\[
5760\div240
=
24\text{ frames/s}
\]

Under this convention, the temporal interval duration is:

\[
\Delta t
=
\frac{16}{24}
=
0.6667\text{ s}
\]

The estimated start and end times for temporal index \(k\) are:

\[
t_{\mathrm{start}}(k)
=
\frac{16k}{24}
\]

\[
t_{\mathrm{end}}(k)
=
\frac{16(k+1)}{24}
\]

These times are relative to the beginning of the corresponding video clip.

RGB I3D feature arrays and VGGish audio feature arrays were verified to have identical temporal sequence length for every inspected video:

\[
T_{\mathrm{RGB}}
=
T_{\mathrm{Audio}}
\]

Thus, the multimodal feature vector at temporal index \(k\) is temporally aligned:

\[
\mathbf{x}_k
=
[
\mathbf{x}^{\mathrm{RGB}}_k,
\mathbf{x}^{\mathrm{Audio}}_k
]
\]

The 16-frame correspondence is directly supported by the official evaluation code. The 24 FPS time conversion is supported by the observed feature-duration relationship; exact encoded-video timestamp verification is deferred until raw video files are inspected using FFmpeg.

### G. Temporal Annotation Mapping

The official XD-Violence annotation file represents anomalous intervals as frame-index pairs:

```text
video_id start_frame end_frame
```

A video may contain multiple anomalous intervals:

```text
video_id
start_1 end_1
start_2 end_2
...
start_m end_m
```

Let the set of annotated anomaly intervals for a video be:

\[
\mathcal{A}
=
\{
[a_1,b_1],
[a_2,b_2],
\ldots,
[a_m,b_m]
\}
\]

The temporal score interval at index \(k\) is:

\[
\mathcal{F}_k
=
[
16k,
16(k+1)-1
]
\]

The temporal anomaly label is assigned by interval overlap:

\[
y_k
=
\begin{cases}
1,
&
\exists[a_j,b_j]\in\mathcal{A}
\text{ such that }
\mathcal{F}_k
\cap
[a_j,b_j]
\neq\varnothing
\\
0,
&
\text{otherwise}
\end{cases}
\]

This labeling procedure makes it possible to compare a model temporal score \(S_k\) with the corresponding official anomaly annotation.

A proof experiment was conducted on five annotation-matched anomalous videos. The proof generated a CSV file containing:

```text
video_id
video_index
temporal_index
start_frame
end_frame
start_time_seconds
end_time_seconds
offline_raw_score
online_raw_score
anomaly_label
```

Among the 800 locally available XD-Violence feature videos, 218 videos matched records in the official temporal annotation file. These 218 videos constitute the valid subset for temporal score-versus-annotation experiments.

### H. Score Calibration

Raw model scores are not automatically calibrated probabilities. Since retention decisions require interpretable and comparable confidence values, the offline video-level score is calibrated using Platt scaling.

Let \(S_{\mathrm{raw}}\) denote the raw offline score. The calibrated event probability is:

\[
p_e
=
\Pr(E=1\mid S_{\mathrm{raw}})
=
\sigma(
\beta_0+
\beta_1S_{\mathrm{raw}}
)
\]

where:

- \(E=1\) denotes a relevant anomalous event.
- \(\beta_0\) is the fitted intercept.
- \(\beta_1\) is the fitted score coefficient.
- \(\sigma(\cdot)\) is the sigmoid function.

The calibration model is fitted only on a stratified validation partition. The locally available 800-video feature partition is divided into:

\[
N_{\mathrm{validation}}
=
160
\]

\[
N_{\mathrm{evaluation}}
=
640
\]

The split is stratified according to the weak video-level label and uses fixed random seed 42.

The validation partition is used only for:

- Calibration model fitting.
- Threshold selection.
- Policy-parameter selection.

The evaluation partition is not used to select calibration coefficients or retention thresholds.

Calibration quality is measured using the Brier score:

\[
\mathrm{BS}
=
\frac{1}{N}
\sum_{i=1}^{N}
(p_i-y_i)^2
\]

where \(p_i\) is the calibrated probability and \(y_i\in\{0,1\}\) is the corresponding weak event label.

Discrimination is measured using average precision and ROC-AUC. The calibrated probability \(p_e\) becomes the event-evidence input for subsequent retention-policy experiments.

### I. Pilot Event-Conditioned Retention Policy

Before introducing operational context, a context-free audio-video event-conditioned retention policy is evaluated as a pilot baseline.

For a video or temporal unit \(u\), let \(p_e(u)\) be its calibrated event probability. The event-conditioned preservation decision is:

\[
P_{\mathrm{event}}(u)
=
\begin{cases}
1,
&
p_e(u)\geq\tau
\\
0,
&
p_e(u)<\tau
\end{cases}
\]

where:

- \(P_{\mathrm{event}}(u)=1\) means preserve.
- \(P_{\mathrm{event}}(u)=0\) means discard.
- \(\tau\) is selected using validation data only.

The threshold is selected to minimize retained video under a required validation evidence-recall target:

\[
\tau^*
=
\arg\min_{\tau}
R_{\mathrm{ret}}(\tau)
\]

subject to:

\[
ER_{\mathrm{validation}}(\tau)
\geq
\eta
\]

where the initial pilot target is:

\[
\eta=0.98
\]

For video-level pilot experiments, evidence recall is defined as:

\[
ER
=
\frac{
\sum_{i=1}^{N}
\mathbf{1}
[
P_i=1
\land
y_i=1
]
}{
\sum_{i=1}^{N}
\mathbf{1}
[
y_i=1
]
}
\]

The retention ratio is:

\[
R_{\mathrm{ret}}
=
\frac{
\sum_{i=1}^{N}
P_i
}{
N
}
\]

Storage reduction is:

\[
SR
=
100
\times
(
1-R_{\mathrm{ret}}
)
\]

The pilot comparison includes:

| Condition | Retention rule |
|---|---|
| C0: Continuous retention | Preserve all videos |
| C2: Audio-video event-conditioned retention | Preserve when \(p_e\geq\tau\) |
| C6: Random matched retention | Preserve the same number of videos as C2, selected randomly |

The random baseline controls for the possibility that retention improvement is caused only by retaining a fixed proportion of videos rather than by informative event scores.

### J. Operational Context Model

Operational context is represented as a transparent external policy profile rather than as a latent visual feature inferred from the surveillance stream.

The context vector is:

\[
\mathbf{c}
=
[
c_{\mathrm{camera}},
c_{\mathrm{scene}},
c_{\mathrm{zone}},
c_{\mathrm{site}},
c_{\mathrm{state}},
c_{\mathrm{time}},
c_{\mathrm{activity}},
c_{\mathrm{risk}},
c_{\mathrm{policy}}
]
\]

where:

| Variable | Description |
|---|---|
| \(c_{\mathrm{camera}}\) | Camera identifier |
| \(c_{\mathrm{scene}}\) | Scene identifier |
| \(c_{\mathrm{zone}}\) | Zone class, such as public, restricted, entry/exit, or corridor |
| \(c_{\mathrm{site}}\) | Site type, such as campus, transit, commercial, industrial, or residential |
| \(c_{\mathrm{state}}\) | Operating state, such as open, closed, restricted-hours, or emergency |
| \(c_{\mathrm{time}}\) | Time period, such as day, evening, or night |
| \(c_{\mathrm{activity}}\) | Expected or allowed activity under the policy |
| \(c_{\mathrm{risk}}\) | Policy-defined risk/sensitivity weight |
| \(c_{\mathrm{policy}}\) | Versioned policy identifier |

Context is not assigned from unknown visual content. For example, a generic video is not labeled “bank,” “school,” or “restricted area” unless this information is documented by dataset metadata or a controlled data-collection protocol.

Because the XD-Violence features do not provide verified operating schedules, site types, restricted-zone labels, or security-policy metadata, XD-Violence is used for audio-video event-score calibration and context-free retention baselines only. Context-conditioned preservation is evaluated separately using a dataset or controlled raw-video collection with documented scene, camera, and operational-policy metadata.

### K. Context-Conditioned Preservation Score

The detector probability and preservation priority are treated as separate quantities.

The calibrated event probability is:

\[
p_e(t)
=
\Pr(
E_t=1
\mid
\mathbf{x}^{\mathrm{RGB}}(t),
\mathbf{x}^{\mathrm{Audio}}(t)
)
\]

A policy-priority function maps context and event type to a normalized priority value:

\[
q(t)
=
g(
\mathbf{c}(t),
\hat{y}(t)
)
\]

where:

\[
q(t)\in[0,1]
\]

The context-conditioned preservation score is:

\[
VAS(t)
=
100
\times
\sigma
\left(
\gamma_0
+
\gamma_1
\operatorname{logit}
(
p_e(t)
)
+
\gamma_2q(t)
+
\gamma_3
\operatorname{logit}
(
p_e(t)
)
q(t)
\right)
\]

where:

\[
\operatorname{logit}(p)
=
\log
\left(
\frac{p}{1-p}
\right)
\]

The interaction term:

\[
\gamma_3
\operatorname{logit}
(
p_e(t)
)
q(t)
\]

models the hypothesis that context should have greater effect when nontrivial event evidence is present.

The VAS is defined as a **preservation-priority score**, not as a probability that an event occurred and not as legal verification of an incident.

The retention decision is:

\[
P(t)
=
\begin{cases}
1,
&
VAS(t)
\geq
\tau_{\mathrm{VAS}}
\\
1,
&
p_e(t)
\geq
\tau_{\mathrm{critical}}
\\
0,
&
\text{otherwise}
\end{cases}
\]

The critical-event override ensures that a high-confidence event is not discarded merely because a local policy profile has low nominal priority.

### L. Temporal Evidence Retention

For a triggered temporal score index \(k\), the initial evidence interval is expanded using pre-event and post-event buffers:

\[
T_k
=
[
t_{\mathrm{start}}(k)-\Delta_{\mathrm{pre}},
t_{\mathrm{end}}(k)+\Delta_{\mathrm{post}}
]
\]

Overlapping or nearby trigger intervals are merged:

\[
\mathcal{I}
=
\operatorname{Merge}
(
\{T_k:
P(k)=1\}
)
\]

where the merge operation combines overlapping intervals and intervals separated by less than a pre-defined temporal gap.

For an annotated event interval:

\[
G_j
=
[
g_j^{\mathrm{start}},
g_j^{\mathrm{end}}
]
\]

event coverage is:

\[
\operatorname{Coverage}
(
G_j,
\mathcal{I}
)
=
\frac{
\left|
G_j
\cap
\left(
\bigcup_{I\in\mathcal{I}}I
\right)
\right|
}{
|G_j|
}
\]

An event is considered preserved if:

\[
\operatorname{Coverage}
(
G_j,
\mathcal{I}
)
\geq
\eta
\]

where \(\eta\) is a pre-specified event-coverage threshold.

The final temporal evidence recall is:

\[
ER_{\mathrm{temporal}}
=
\frac{
\sum_{j=1}^{M}
\mathbf{1}
[
\operatorname{Coverage}
(
G_j,
\mathcal{I}
)
\geq
\eta
]
}{
M
}
\]

where \(M\) is the number of ground-truth event intervals.

Temporal clip extraction, actual video-byte retention, encryption, hashing, IPFS storage, Hyperledger Fabric recording, retrieval, and tampering verification are executed only after raw video files and exact encoded-video timing are verified.

***

## Important wording to retain

Use this statement in the paper:

> The pretrained audio-visual detector is not presented as a novel contribution. It is used as a fixed temporal event-score generator for evaluating the proposed evidence-preservation policy.

Use this statement too:

> The 16-frame temporal correspondence is directly supported by the official evaluation implementation. Time conversion under the 24 FPS convention is feature-derived and requires raw-video inspection before being described as exact encoded-video timing.

Do **not** yet include numerical AP, ROC-AUC, Brier, retention, or storage-reduction values in this methodology section. Those belong in Results only after the final experiment protocol is frozen.