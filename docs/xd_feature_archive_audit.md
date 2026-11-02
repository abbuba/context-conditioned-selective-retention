# XD-Violence Feature Archive Audit

**Audit time (UTC):** 2026-10-02T07:16:14.243296+00:00

## Available Disk Space

| Field | Value |
|---|---:|
| Total disk capacity (GB) | 451.84 |
| Used disk capacity (GB) | 329.89 |
| Free disk capacity (GB) | 121.94 |

## Archives Detected

| Archive | Compressed GB | Uncompressed GB | `.npy` files | RGB | Flow | Audio | SHA-256 |
|---|---:|---:|---:|---|---|---|---|
| `i3d-features.zip` | 38.348 | 44.545 | 47540 | True | True | False | `44a324094ae693d87dde2fdb1282cf42968ce6dec50cb379bff595af1db72d69` |
| `vggish-features.zip` | 0.225 | 0.557 | 4754 | True | False | True | `ca414a0d849754e3a80bed672843e05670ce67b93ae9ee36c4fd1113f5865816` |

## Extraction Decision

The selected XDVioDet checkpoint operates in MIX2 mode. Only I3D RGB and
VGGish audio features are required. Optical-flow features must not be extracted
unless a future documented ablation explicitly requires them.

The next extraction script will extract only:
1. I3D RGB feature files.
2. VGGish audio feature files.
3. Required test-list and annotation resources.

## Sample Archive Contents

### i3d-features.zip
```text
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A__0.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A__1.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A__2.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A__3.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A__4.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-40-52_00-42-01_label_A__0.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-40-52_00-42-01_label_A__1.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-40-52_00-42-01_label_A__2.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-40-52_00-42-01_label_A__3.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-40-52_00-42-01_label_A__4.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-50-26_00-55-16_label_A__0.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-50-26_00-55-16_label_A__1.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-50-26_00-55-16_label_A__2.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-50-26_00-55-16_label_A__3.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#00-50-26_00-55-16_label_A__4.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A__0.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A__1.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A__2.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A__3.npy
i3d-features/FlowTest/A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A__4.npy
i3d-features/FlowTest/About.Time.2013__#00-23-50_00-24-31_label_A__0.npy
i3d-features/FlowTest/About.Time.2013__#00-23-50_00-24-31_label_A__1.npy
i3d-features/FlowTest/About.Time.2013__#00-23-50_00-24-31_label_A__2.npy
i3d-features/FlowTest/About.Time.2013__#00-23-50_00-24-31_label_A__3.npy
i3d-features/FlowTest/About.Time.2013__#00-23-50_00-24-31_label_A__4.npy
i3d-features/FlowTest/About.Time.2013__#00-30-50_00-32-31_label_A__0.npy
i3d-features/FlowTest/About.Time.2013__#00-30-50_00-32-31_label_A__1.npy
i3d-features/FlowTest/About.Time.2013__#00-30-50_00-32-31_label_A__2.npy
i3d-features/FlowTest/About.Time.2013__#00-30-50_00-32-31_label_A__3.npy
i3d-features/FlowTest/About.Time.2013__#00-30-50_00-32-31_label_A__4.npy
```
### vggish-features.zip
```text
vggish-features/test/A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A__vggish.npy
vggish-features/test/A.Beautiful.Mind.2001__#00-40-52_00-42-01_label_A__vggish.npy
vggish-features/test/A.Beautiful.Mind.2001__#00-50-26_00-55-16_label_A__vggish.npy
vggish-features/test/A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A__vggish.npy
vggish-features/test/About.Time.2013__#00-23-50_00-24-31_label_A__vggish.npy
vggish-features/test/About.Time.2013__#00-30-50_00-32-31_label_A__vggish.npy
vggish-features/test/About.Time.2013__#00-40-52_00-42-31_label_A__vggish.npy
vggish-features/test/Bad.Boys.1995__#00-10-00_00-11-52_label_A__vggish.npy
vggish-features/test/Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6__vggish.npy
vggish-features/test/Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0__vggish.npy
vggish-features/test/Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0__vggish.npy
vggish-features/test/Bad.Boys.II.2003__#01-11-16_01-14-00_label_A__vggish.npy
vggish-features/test/Be.with.You.2018__#00-04-20_00-05-35_label_A__vggish.npy
vggish-features/test/Be.with.You.2018__#00-16-18_00-18-09_label_A__vggish.npy
vggish-features/test/Be.with.You.2018__#00-23-50_00-24-31_label_A__vggish.npy
vggish-features/test/Be.with.You.2018__#01-14-30_01-16-59_label_A__vggish.npy
vggish-features/test/Be.with.You.2018__#01-42-06_01-45-53_label_A__vggish.npy
vggish-features/test/Before.Sunrise.1995__#00-03-00_00-04-05_label_A__vggish.npy
vggish-features/test/Before.Sunrise.1995__#00-04-20_00-05-35_label_A__vggish.npy
vggish-features/test/Before.Sunrise.1995__#00-23-50_00-24-31_label_A__vggish.npy
vggish-features/test/Before.Sunrise.1995__#00-40-52_00-42-01_label_A__vggish.npy
vggish-features/test/Before.Sunrise.1995__#00-42-52_00-45-31_label_A__vggish.npy
vggish-features/test/Before.Sunrise.1995__#01-36-45_01-38-10_label_A__vggish.npy
vggish-features/test/Before.Sunset.2004__#00-23-50_00-24-31_label_A__vggish.npy
vggish-features/test/Before.Sunset.2004__#01-14-30_01-16-51_label_A__vggish.npy
vggish-features/test/Black.Hawk.Down.2001__#00-06-44_00-07-39_label_A__vggish.npy
vggish-features/test/Black.Hawk.Down.2001__#01-13-59_01-14-49_label_B2-0-0__vggish.npy
vggish-features/test/Black.Hawk.Down.2001__#01-32-40_01-34-00_label_B4-0-0__vggish.npy
vggish-features/test/Black.Hawk.Down.2001__#01-36-44_01-37-33_label_A__vggish.npy
vggish-features/test/Black.Hawk.Down.2001__#01-42-58_01-43-58_label_G-0-0__vggish.npy
```
