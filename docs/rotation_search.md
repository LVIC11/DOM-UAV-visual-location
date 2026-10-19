# Physical Query Rotation And Satellite Matching

`scripts/main.py --query-rotation-cw` describes a rotation that has ALREADY
been applied. It does not physically rotate input images. Do not pass an angle
for unchanged images and assume that SuperPoint receives a rotated image.

The independent `scripts/match_rotated_queries.py` entry point actually rotates
the original image BEFORE SuperPoint extraction, then uses the existing SVL
SuperGlue and planar-PnP implementation by default. Historical source archives
and frozen experiments are preserved. This entry point supports arbitrary
clockwise angles, expanded bounds without cropping, and explicit affine inverse
mapping through any query resize. Camera calibration stays in original pixels.

## Run

From the repository root, using the existing environment:

```bash
/home/sy/.cache/pypoetry/virtualenvs/svl-FMcZIkqU-py3.10/bin/python \
  scripts/match_rotated_queries.py \
  --image-folder data/my_experiment/query_test/winter_50 \
  --map-db data/my_experiment/map/tiles \
  --camera-calib /media/sy/30C2CCDEC2CCAA04/datasets/a_low/winter/nav_left.yaml \
  --query-sizes 480,640,800 --keypoint-threshold 0.005 \
  --pnp-homography-prefilter --output NEW_OUTPUT_DIRECTORY
```

The source folder must contain original 1280x720 camera images and metadata.csv
with filename/timestamp. GPS columns are NOT used for matching, selecting tiles,
choosing rotations or passing geometric gates. All available tiles are searched.
The source images are never overwritten. Existing output directories are refused.

For the user-provided winter_50 angles, use the fixed policy instead:

```bash
/home/sy/.cache/pypoetry/virtualenvs/svl-FMcZIkqU-py3.10/bin/python \
  scripts/match_rotated_queries.py \
  --image-folder data/my_experiment/query_test/winter_50 \
  --map-db data/my_experiment/map/tiles \
  --rotation-ranges '4-5:270,6-15:90,16-22:180,23-65:270,66-75:90,76-87:270' \
  --query-sizes 480,640,800,1280 --keypoint-threshold 0.005 \
  --pnp-homography-prefilter --output NEW_FIXED_OUTPUT_DIRECTORY
```

Ranges refer to source filename numbers, with inclusive endpoints. Angles are
clockwise; counterclockwise 90 degrees is clockwise 270 degrees. Fixed mode
disables ALL automatic angle search/refinement and checks only query sizes at
the prescribed angle. Every selected source image must have exactly one angle.
All 84 full-resolution rotations are saved, even when matching or PnP fails;
user-prescribed orientation is distinct from geometric localization acceptance.
Unrotated matching is not run in this mode, so the report marks this baseline
as untested, not zero. The original/rotated montage compares image pixels only.

Default angle search: 0/90/180/270 plus the remaining 30-degree grid. Promising
angles receive a +/-15 degree search at 5-degree spacing; strong but rejected
candidates can receive a further +/-5 degree search at 1-degree spacing.
Other query scales are checked near the best angle. This is a bounded heuristic,
not exhaustive search of all angles/scales/tiles simultaneously, and it does not
establish uniqueness in repetitive terrain. Use `--angles`, `--fine-radius`,
`--fine-step`, `--refine-step`, `--frames` or `--max-images` for diagnostics.

`--pnp-homography-prefilter` feeds the image/satellite homography consensus into
PnP. It is OFF by default to make this policy change explicit. PnP still requires
>=20 inliers, >=0.35 consensus, <=6 px reprojection for inliers, >=0.03 image
bounding-box coverage, positive depth, camera height in [5,1000] m and <=30 deg
optical tilt. With this option the 0.35 ratio is measured AFTER homography
prefiltering, not against all raw SuperGlue matches. Both denominators and counts
are exported. This changes the input admission policy even though numeric PnP
thresholds are unchanged; do not attribute all gains to rotation alone.

The optional detector threshold 0.005 is SuperPointConfig's default, versus
main.py's 0.01. SuperGlue outdoor weights, confidence threshold 0.5, 20 Sinkhorn
iterations and unbounded SuperPoint count otherwise follow main.py. The final
geometric test is mandatory; more raw matches alone are not usable constraints.

## Output

- `rotation_manifest.json`: every processed image, native timestamp, all tested
  angles/scales/tiles, stage diagnostics, chosen angle, affine transforms,
  PnP verification flag, input SHA256 and processing policy.
- `images_rotated/*.png`: full-resolution physical rotations. In automatic mode,
  unverified candidates are saved for review, NOT declared correct north-up orientations.
  If no homography candidate exists, angle is unresolved and no arbitrary
  rotated image is invented. Its frame remains in the manifest. In fixed mode,
  all requested images are saved at the user-prescribed angles regardless of
  whether geometry is verified.
- `images_rotated/metadata.csv`: rotated filenames, original timestamps, chosen
  angles and verification flags. Arbitrary-angle results must use their affine
  mapping; the older main.py cardinal-only inverse mapping is not interchangeable.
- `visualizations_rotated`: matching-input image versus satellite tile, with
  actual rotation and PnP/tentative labels.
- `visualizations_original`: the same correspondences drawn in original pixels.
- `matched_keypoints.json`: compatible successful-record list for downstream
  SVL exporters. Contains ONLY PnP inlier points with satellite lat/lon,
  original `query_x/query_y`, transformed matching-input pixels, native timestamp,
  local feature indices, full PnP result and inverse affine matrix.
- `source_archive`: code snapshot for this run. An interrupted diagnostic is
  preserved as failed, not passed off as a completed full batch.

`query_idx` is local to this extraction, not an AIRSLAM feature/map-point ID.
Returned image pixels are original distorted left-camera coordinates. AIRSLAM
rectification and exact source-frame association must be verified separately
before BA ingestion. Arbitrary-angle inverse mapping and the PnP subset mask are
tested to preserve the correspondence to original feature indices.

## Review

```bash
/home/sy/.cache/pypoetry/virtualenvs/svl-FMcZIkqU-py3.10/bin/python \
  scripts/report_rotation_search.py COMPLETED_OUTPUT_DIRECTORY
```

Creates `frame_summary.csv`, `rotation_summary.png`, `before_after.png`,
`position_reference_audit.json` and `summary.json`. Input hashes, timestamps,
raw-pixel bounds, inverse mappings and inlier-only export are verified. Metadata
GPS is opened for post-hoc reference distances only, not filtering/reweighting.
The camera/RTK antenna lever arm is unmodelled. PnP acceptance is a geometric
check, not a certificate of correct geolocation or improved SLAM accuracy.
No AIRSLAM node, graph insertion, bag playback or online feedback is performed.

## Exact Camera Header Time

The supplied winter_50 metadata timestamps are bag RECORD times, not image
header times. All 84 records were associated exactly to `/camera/left/image_raw`
and their JPEG/raw content checked. The maximum record/header offset is about
18.624 ms; source images and metadata are not rewritten.

Use the separate ROS-capable system Python to recover headers, and retime a
completed match list into a NEW file without changing any pixels or geodata:

```bash
python3 scripts/associate_query_headers.py associate \
  --image-folder data/my_experiment/query_test/winter_50 \
  --bag /media/sy/30C2CCDEC2CCAA04/datasets/a_low/winter/50_10.bag \
  --output NEW_HEADER_ASSOCIATIONS
python3 scripts/associate_query_headers.py retime \
  --input COMPLETED_OUTPUT/matched_keypoints.json \
  --associations NEW_HEADER_ASSOCIATIONS/associations.json \
  --output COMPLETED_OUTPUT/matched_keypoints_header_time.json
```

Original match timestamps remain available as `metadata_timestamp` and
`bag_record_timestamp`; `timestamp` in the new file is the exact image header.
This is an exact record association, not snapping to a nearby SLAM keyframe.
The image still needs to exist in the optimization graph/window before use.

## Completed Fixed-Angle Batch, 2026-10-16

`data/my_experiment/output_winter50_fixed_rotation_20261016` links to
`/media/sy/data2/AIRSLAM_results/experiments/winter50_rotation_20261016/fixed_user_angles_r1`.
All 84 images were physically rotated at the user's fixed angles and matched
against all 18 tiles at four scales. The original images remain unchanged.
Only 000070 and 000071 passed current PnP gates, with 23 and 25 inliers. All
rotations passed pixel-exact checks; 110 input hashes and inverse pixels passed.
Header-time output preserves all 48 inlier points. Earlier automatic output
`full_r2` is diagnostic and superseded, not this fixed-angle batch.

This task is SVL offline prematching only: no AIRSLAM node, BA insertion or bag
playback. PnP is SVL geometry validation. Rejection diagnostics remain available;
do not describe the other 82 frames as unprocessed or as accepted constraints.

## Plot Saved Predictions

This plotting entry reads existing results; it never reruns matching or AIRSLAM:

```bash
/home/sy/.cache/pypoetry/virtualenvs/svl-FMcZIkqU-py3.10/bin/python \
  scripts/plot_prematch_trajectory.py \
  --metadata data/my_experiment/query_test/winter_50/metadata.csv \
  --matches data/my_experiment/output_winter50_fixed_rotation_20261016/matched_keypoints_header_time.json \
  --tiles data/my_experiment/map/tiles --output NEW_PLOT_DIRECTORY
```

The actual matching tiles are stitched using their map.csv geography, not the
legacy mosaic script's hardcoded bounds. Green is the full metadata reference
track; magenta is accepted PnP camera positions, with frame labels and horizontal
reference distances. Lines connect adjacent accepted source frames only, with
a maximum five-second gap; rejected frames are not interpolated. GPS is used
only for visualization/reference errors, never to align or filter predictions.

Completed output is `output_winter50_fixed_rotation_20261016/trajectory_visualization_r2`:
`trajectory.jpg`, `trajectory_zoom.jpg`, `trajectory.csv` and `summary.json`.
Aliases in `data/my_experiment` are `trajectory_winter50_planar_pnp.jpg` and
`trajectory_winter50_planar_pnp_zoom.jpg`. Two accepted frames have a horizontal
reference RMSE of 5.5641 m, not a full-trajectory accuracy estimate. The earlier
plot r1 stopped during summary serialization and is diagnostic only.

## Legacy Homography Image-Center Prediction

For an explicit no-PnP comparison on the SAME fixed rotations, add
`--position-method homography_center` to `match_rotated_queries.py` and omit
`--pnp-homography-prefilter`. Keep the same four query sizes and detector threshold:

```bash
/home/sy/.cache/pypoetry/virtualenvs/svl-FMcZIkqU-py3.10/bin/python -u \
  scripts/match_rotated_queries.py \
  --image-folder data/my_experiment/query_test/winter_50 \
  --map-db data/my_experiment/map/tiles \
  --rotation-ranges '4-5:270,6-15:90,16-22:180,23-65:270,66-75:90,76-87:270' \
  --query-sizes 480,640,800,1280 --keypoint-threshold 0.005 \
  --position-method homography_center --output NEW_HOMOGRAPHY_OUTPUT
```

This physically rotates pixels before SuperPoint, retains SuperGlue and H-RANSAC,
and projects the processed query center `(width/2, height/2)` through H. The
historical integer satellite pixel is interpolated into tile latitude/longitude.
Only finite centers inside the satellite tile are accepted. Candidate ranking
uses raw SuperGlue match count, as in the previous image-center predictor.
PnP is NEVER called. The default minimum of four H inliers restores the legacy
weak gate, not the 20-inlier PnP gate; `--homography-min-inliers` is explicit.

Outputs are labeled `position_method=homography_center` and represent the image
ground footprint, NOT a solved 3D camera/body position. Only H inlier points are
exported, `is_pnp_inlier` is false, and `planar_pose` is null. Exact inverse raw
pixels and separate header-time retiming remain available. The PnP-only AIRSLAM
exporter refuses these records instead of silently labeling them as PnP factors.
No AIRSLAM execution or BA is part of this comparison.

Use `plot_prematch_trajectory.py` for its metadata-reference CSV/statistics and
trajectory plots; it detects the method automatically. The older
`report_rotation_search.py` currently validates PnP-only point masks and is NOT
the homography report entry. More outputs under a four-point gate can include
false matches or unstable extrapolation; both all-output and common-frame errors
must be compared, without GT-based removal or final trajectory alignment.

PnP remains the default for existing commands. Historical PnP output/data and
source archives are preserved; the live matcher scripts have since evolved, so
their old input source hashes must be checked against those archived versions.

The usual `scripts/main.py` also accepts `--position-method homography_center`
and `--homography-min-inliers 4`. Its image folder must ALREADY contain rotated
images, and `--query-rotation-ranges` declares their angles for inverse pixel
export; it still does not physically rotate images. This does not change the
other entry's explicit preprocessing or the defaults of older PnP commands.

Completed H-center comparison: `data/my_experiment/output_winter50_homography_center_20261016`.
All 84 images were processed; 26 predictions / 571 H inlier points were exported.
All 6048 attempt-level feature/raw/H-inlier counts agree with the PnP control.
H-center reference RMSE is 73.2817 m on its 26 outputs (median 11.0031 m),
with 271.0796 m maximum. On common 000070/000071, H RMSE is 5.4621 m versus
PnP 5.5641 m. Different all-output cohorts must not be called a paired comparison.
See `comparison_with_pnp.json`, `verification.json` and the result README.
The extra coverage includes weak matches/outliers; do not equate 26 predictions
with 26 reliable camera-position observations or improved SLAM accuracy.
