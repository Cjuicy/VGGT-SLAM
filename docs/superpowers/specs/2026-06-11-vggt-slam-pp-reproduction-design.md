# VGGT-SLAM++ Reproduction Design

Date: 2026-06-11

## 1. Purpose

This project will reproduce VGGT-SLAM++ incrementally on top of the provided
VGGT-SLAM 1.0 source while keeping the baseline understandable and minimally
modified.

The implementation must support:

- Manual review at every milestone.
- Offline replay of every algorithm stage.
- Explicit separation between paper-defined behavior, ported source behavior,
  and experimental assumptions.
- Local development on the existing `vggt-dem` Conda environment on macOS.
- Full inference and performance evaluation on AutoDL CUDA machines.
- Git-based transfer without committing datasets, weights, caches, or other
  large generated artifacts.

The initial implementation target is not the entire online system. It is a
reviewable baseline and data contract, followed by an independently testable
DEM pipeline.

## 2. Evidence And Reproduction Labels

Every nontrivial algorithm or configuration field must carry one of these
labels in documentation and experiment metadata:

- `paper_exact`: The paper and appendix provide enough detail to implement the
  behavior directly.
- `source_ported`: The behavior is migrated from a cited external
  implementation placed under `external_sources/`.
- `experimental`: The paper leaves an implementation gap and the project uses
  a replaceable hypothesis.

An `experimental` implementation must not be presented as the authors'
original implementation.

## 3. Architecture Decision

Use a sidecar extension package.

```text
VGGT-SLAM-version1.0/    Frozen baseline snapshot and minimal export bridge
vggt_slam_pp/            New VGGT-SLAM++ implementation
configs/                 Dataset and runtime profiles
environment/             Local macOS and AutoDL CUDA environment definitions
external_sources/        Ignored third-party repos plus tracked provenance
artifacts/               Replay caches and generated outputs, ignored by Git
tests/                   Synthetic and integration tests
docs/                    Design, algorithm notes, provenance, and runbooks
```

`VGGT-SLAM-version1.0/` remains the baseline. New DEM, retrieval, VPR, and
Sim(3) backend code must not be added to the baseline package.

The baseline may receive a small export hook only when required data cannot be
obtained without one. Each baseline modification must document:

- Why the modification is required.
- Which paper stage consumes the output.
- Input and output shapes, units, and coordinate frames.
- Why the change does not alter baseline estimation behavior.

## 4. Proposed Repository Structure

```text
.
├── VGGT-SLAM-version1.0/
├── vggt_slam_pp/
│   ├── contracts/
│   │   ├── submap.py
│   │   ├── scale.py
│   │   ├── dem.py
│   │   ├── retrieval.py
│   │   └── graph.py
│   ├── adapters/
│   │   └── vggt_slam_v1.py
│   ├── geometry/
│   │   ├── depth_filter.py
│   │   ├── plane.py
│   │   ├── canonical_frame.py
│   │   └── sim3_measurement.py
│   ├── dem/
│   │   ├── lattice.py
│   │   ├── rasterizer.py
│   │   ├── reducers.py
│   │   ├── normalization.py
│   │   └── visualization.py
│   ├── embedding/
│   │   ├── dinov2.py
│   │   ├── weighting.py
│   │   └── signatures.py
│   ├── retrieval/
│   │   ├── hnsw.py
│   │   ├── voting.py
│   │   └── candidate_graph.py
│   ├── vpr/
│   │   ├── anyloc.py
│   │   └── verifier.py
│   ├── backend/
│   │   ├── interface.py
│   │   ├── sim3_graph.py
│   │   └── scheduler.py
│   └── cli/
├── configs/
│   ├── profiles/
│   │   ├── generic_relative.yaml
│   │   ├── tum_indoor.yaml
│   │   └── kitti_outdoor.yaml
│   └── runtime/
│       ├── local_cpu.yaml
│       └── autodl_cuda.yaml
├── environment/
│   ├── local-macos.yml
│   ├── autodl-cuda.yml
│   └── compatibility.md
├── external_sources/
│   ├── manifest.yaml
│   ├── README.md
│   └── migration_notes/
├── artifacts/
│   ├── submaps/
│   ├── dem/
│   ├── descriptors/
│   ├── indices/
│   ├── graph/
│   └── runs/
├── tests/
│   ├── synthetic/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── algorithms/
    ├── provenance/
    └── runbooks/
```

The exact file count may change during implementation. Module ownership and
interfaces must remain clear.

## 5. Runtime Environments

### 5.1 Local macOS

Use the existing Conda environment:

```text
vggt-dem
Python 3.11.15
PyTorch 2.11.0
CPU available
MPS currently unavailable
```

The local CPU path is a required compatibility target for:

- Synthetic geometry tests.
- DEM construction and visualization.
- Small DINOv2 descriptor tests.
- FAISS-HNSW correctness and recall tests.
- Sim(3) math tests independent of the final optimizer binding.

Device selection is `CUDA -> MPS -> CPU`, but CPU behavior must remain valid.
MPS is an optional acceleration path and is not a completion requirement.

### 5.2 AutoDL CUDA

AutoDL is responsible for:

- Full VGGT inference.
- Baseline sequence runs.
- Full-resolution DEM generation.
- Large descriptor batches.
- End-to-end evaluation and performance measurement.

Local and AutoDL configurations and run records remain separate. Code must not
contain machine-specific absolute paths.

## 6. Milestones And Review Gates

### M0: Baseline Freeze And Cloud Validation

1. Record baseline source provenance and archive SHA-256.
2. Track the baseline core source in Git.
3. Run the `office_loop` smoke test on AutoDL.
4. Run TUM `freiburg1_desk` as the first formal baseline.
5. Export a reviewable submap cache.
6. Record image, CUDA, Python, dependency, weight, command, and output details.

The current `main_offline.py` is not the cache contract:

- `--save_only` currently bypasses the save call.
- Cached data omits replay-critical loop metadata.
- Its comments claim images can be omitted, while `Solver.add_points()` reads
  `pred_dict["images"]`.

M0 therefore adds a minimal exporter rather than extending that temporary
script.

### M1: Scale-Aware DEM

1. Validate the submap cache schema.
2. Filter confidence and depth outliers.
3. Propagate submap-relative scale into a common map-unit frame.
4. Fit and validate a dominant plane.
5. Construct and freeze a canonical DEM frame.
6. Construct a stable metric or relative tile lattice.
7. Rasterize mean, max, and softmax reducers.
8. Generate grayscale and human visualization artifacts.
9. Validate synthetic and real submap examples.

M1 is complete only after manual review of code, formulas, statistics, and
rendered DEM artifacts.

### M2: DINOv2 And Covisibility Candidates

1. Produce weighted DINOv2 tile and query descriptors.
2. Build exact-search and FAISS-HNSW retrieval backends.
3. Compare HNSW recall against exact retrieval.
4. Aggregate tile matches to submap candidates.
5. Persist raw chip-to-tile evidence and aggregated rankings.

### M3: VPR And Geometric Edge Measurement

1. Port and validate AnyLoc behavior from a supplied external source.
2. Refine candidates only within the M2 covisibility window.
3. Convert accepted retrieval evidence into independent geometric evidence.
4. Estimate or reject a relative Sim(3) measurement.
5. Record descriptor scores, geometric residuals, degeneracy checks, and edge
   provenance.

### M4: Spatially Corrective Backend

1. Validate Sim(3) group math and synthetic graph recovery.
2. Implement a backend-independent `LocalSim3Backend` interface.
3. Implement or bind a reliable 7-DoF optimizer.
4. Optimize a bounded covisibility subgraph.
5. Correct poses and invalidate affected map products.
6. Add online scheduling only after deterministic offline replay is correct.

## 7. Submap Cache Contract

Each submap has an independent directory:

```text
artifacts/submaps/<run_id>/<submap_id>/
├── metadata.json
├── images.npz
├── cameras.npz
├── geometry.npz
├── confidence.npz
└── checksums.json
```

`metadata.json` includes:

- Schema version.
- Submap and frame identifiers.
- Source dataset and source paths relative to a configured root.
- Array shapes and dtypes.
- Transform direction conventions.
- Camera convention.
- Coordinate frame name.
- Unit and `scale_status`.
- `meters_per_map_unit` when known.
- Scale anchor source and provenance.
- Baseline commit/archive hash.
- Weight hashes.
- Device and runtime identity.
- Algorithm profile and evidence labels.

Required geometry data includes:

- Camera poses and intrinsics.
- Dense points or replayable depth.
- Confidence maps.
- Selected frame identifiers.
- Temporal submap relationship.
- Relative scale estimates and their provenance.

Loop information must be serialized structurally, not reduced to a count.

## 8. Scale Policy

The project distinguishes:

```text
scale_status = relative | metric
```

For submap point `p_i`:

```text
p_world = s_i R_i p_i + t_i
p_metric = gamma * p_world
```

where `gamma` has units `meter / map-unit`.

Sim(3) odometry can produce a common relative map scale, but pure monocular
geometry does not uniquely determine `gamma`. A metric label therefore
requires a documented anchor.

Possible anchors are dataset- and experiment-dependent and are not selected
in this design:

- Sensor depth.
- Known camera height or baseline.
- External odometry.
- A source implementation's documented metric policy.
- Ground truth for diagnostics only, never silently used in a claimed
  uncalibrated reproduction.

Without a valid metric anchor:

- Use `generic_relative`.
- Express resolution as map-units per pixel.
- Use an explicit `tile_size_map_units`.
- Mark results `experimental`.
- Do not claim 2-by-2 meter tiles.

## 9. Dataset Profiles

Scene-dependent behavior is configuration, not branching algorithm code.

Profiles include:

- Scale policy and anchor.
- Confidence policy.
- Minimum and maximum depth.
- Plane RANSAC threshold and iteration count.
- Minimum plane inlier ratio.
- Canonical frame policy.
- DEM resolution policy.
- Tile size.
- Reducer and temperature.
- Normalization percentiles.
- Edge enhancement policy.
- DINO input size and context.
- HNSW parameters.
- Retrieval and VPR thresholds.

Initial profiles:

- `generic_relative`: correctness testing without metric claims.
- `tum_indoor`: values determined during TUM validation.
- `kitti_outdoor`: values determined during KITTI validation.

Unknown profile values remain explicit validation tasks. They must not be
hidden behind undocumented defaults.

## 10. DEM Mathematics And Coordinate Rules

### 10.1 Plane And Canonical Frame

For points in one common world and scale version:

```text
Pi = { p in R^3 : n^T p + d = 0 }                         (paper 9)
R = [x y z] in SO(3), z = n                               (paper 10)
p_tilde_i = R^T (p_i - o) = (u_i, v_i, h_i)               (paper 11)
```

RANSAC identifies plane inliers and SVD/PCA refines the plane and in-plane
axes.

Additional reproduction rules:

- Resolve plane normal sign deterministically.
- Resolve PCA axis sign deterministically.
- Detect near-equal in-plane eigenvalues.
- Freeze the canonical frame after gallery initialization, or explicitly
  align and version a replacement frame.
- Record `T_world_dem`.

### 10.2 Grid Resolution

```text
S = max(u_1 - u_0, v_1 - v_0)                             (paper 15)
mpp = S / target_px_long                                  (paper 16)
W_px = ceil((u_1 - u_0) / mpp)
H_px = ceil((v_1 - v_0) / mpp)                            (paper 17)
```

In relative mode, `mpp` means map-units per pixel.

The paper simultaneously mentions dynamic `mpp`, fixed `tile_px`, and
2-by-2 meter tiles. These constraints are not generally all satisfiable.
The initial design treats physical tile size as the primary metric constraint:

```text
tile_px = round(tile_size_m / mpp)
tile_size_actual = tile_px * mpp
```

The quantization error is recorded. This policy remains replaceable if source
code shows another interpretation.

The paper's "90k pixels" and "4096 spatial tiles" are not sufficiently
specified and remain profile-validation tasks.

### 10.3 Stable Lattice

The paper derives indices from a bounding-box origin. An online implementation
must prevent tile identities from shifting as the map grows.

- Freeze `lattice_origin`, `mpp`, and `tile_px` when the gallery is initialized.
- Use signed integer tile coordinates.
- Treat the current bounding box as occupancy metadata, not tile identity.

```text
x_hat_i = (u_i - lattice_u0) / mpp                         (paper 19)
y_hat_i = (v_i - lattice_v0) / mpp
I_u = floor(x_hat_i / tile_px)                             (paper 20)
I_v = floor(y_hat_i / tile_px)                             (paper 21)
x = round(x_hat_i - I_u * tile_px)                         (paper 22)
y = round(y_hat_i - I_v * tile_px)                         (paper 23)
```

### 10.4 Height Reducers

For heights in one pixel:

```text
H(x, y) = reducer({h_k})                                   (paper 25)
mean = sum(h_k) / K                                        (paper 26)
max = max(h_k)                                             (paper 27)
softmax = sum(exp(h_k/tau) h_k) / sum(exp(h_k/tau))        (paper 28)
```

The paper's default softmax temperature is `tau = 0.02`.
Numerically stable implementation subtracts the maximum logit before
exponentiation.

### 10.5 Grayscale Normalization

```text
h_min = percentile_0.5(H)
h_max = percentile_99.5(H)                                (paper 29)
I_0 = (clip(H, h_min, h_max) - h_min) / (h_max - h_min)   (paper 30)
```

Empty pixels retain a mask and render white for inspection.

Gallery normalization statistics are frozen or versioned. Queries must use
the corresponding gallery statistics. If statistics change, affected
descriptors and indices are rebuilt.

The paper does not clearly state how edge enhancement is combined with `I_0`.
This remains an evidence-labelled strategy.

## 11. DINOv2 Embedding

The likely base weight is:

```text
weights/dinov2_vitb14_pretrain.pth
SHA-256: 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

ViT-B/14 is consistent with the paper's 768-dimensional descriptor.
The model is frozen.

The existing `dino_salad.ckpt` is baseline SALAD retrieval state and is not
treated as VGGT-SLAM++ AnyLoc evidence.

Paper equation:

```text
v_k = sum_j(w_j m_j f_theta(p_j)) / sum_j(w_j m_j)        (paper 1)
```

Known behavior:

- Global tile descriptor uses a centered 9-by-9 tile neighborhood.
- `w_j` is a Gaussian positional weight.
- `m_j` is derived from DEM gradient magnitude.
- Flat or low-information regions receive less weight.
- Descriptors are normalized.

Unknown behavior:

- DINO layer and facet.
- Input image size.
- Gaussian sigma.
- Gradient-to-visibility mapping.
- Missing-neighbor padding.
- Exact query-chip aggregation region.

Physical DINO patch footprint is explicit:

```text
context_size = context_tiles * tile_size
dino_mpp = context_size / dino_input_px
patch_footprint = 14 * dino_mpp
```

Changing DINO input size changes physical receptive field and therefore
changes the descriptor signature.

## 12. M2 Retrieval

Normalize descriptors and compute:

```text
s(chi_q, tau_k) =
    v_q^T v_k / (||v_q|| ||v_k||)                          (paper 2)
```

Use exact search as the correctness oracle. Use FAISS-HNSW as the scalable
implementation.

For each query chip, persist:

- Retrieved tile identifiers.
- Parent submap identifiers.
- Raw similarity scores.
- Rank.
- Index signature and parameters.

Paper submap voting:

```text
Score(S) += sum_{tau_k in S} s(chi_q, tau_k)               (paper 3)
```

Top `K = 10` submaps are candidate covisible neighbors.

Raw-sum voting is implemented as `paper_exact`. Normalized voting may be
evaluated only as `experimental`, because raw sums can favor submaps with more
retrieved tiles.

Recent temporal neighbors, duplicate matches, and score thresholds remain
profile settings because the paper does not specify them.

## 13. M3 VPR And Relative Sim(3)

AnyLoc operates only inside the M2 covisibility window.

Separate interfaces:

```text
VprRefiner.rank(query, candidates) -> ScoredTileMatches
RelativeSim3Estimator.verify(matches, geometry)
    -> Sim3Measurement | Rejection
```

Retrieval scores are not geometric transformations.

The paper does not specify:

- AnyLoc layer, facet, VLAD vocabulary, or cluster count.
- How chip-to-tile matches map to 3D correspondences.
- How the relative `T_hat_ij` is estimated.
- Acceptance thresholds and degeneracy checks.

The final geometric verifier must check:

- Sufficient independent correspondences.
- Non-collinearity and non-degenerate 3D support.
- Valid scale range.
- Forward/backward consistency.
- 3D alignment residuals.
- Reproducible rejection reasons.

Each edge has a lifecycle:

```text
proposed -> verified -> active
         -> rejected
active   -> stale
```

## 14. M4 Sim(3) Backend

The paper optimizes:

```text
min_{T_i in Sim(3)}
sum_(i,j in E) || Log_Sim3(T_j^-1 T_i T_hat_ij) ||^2_{Sigma_ij}
                                                               (paper 4)
```

State:

```text
T_i = (s_i, R_i, t_i) in Sim(3)
```

The paper describes a GTSAM optimizer on the Sim(3) manifold. The local
`vggt-dem` GTSAM package exposes `Similarity3`, but Python `Values`,
`PriorFactorSimilarity3`, and `BetweenFactorSimilarity3` bindings are not
available. The optimizer implementation therefore remains behind:

```text
LocalSim3Backend.optimize(active_subgraph) -> CorrectedSubmapPoses
```

Possible implementations to investigate later:

- The authors' GTSAM branch or bindings.
- A small C++ GTSAM extension.
- A verified alternative Sim(3) optimizer.

Do not fall back to Pose3 while calling the result Sim(3).

The paper calls this local bundle adjustment, but equation (4) shows a
motion-only Sim(3) pose graph rather than explicit point or camera observation
variables. The project will use the precise term "local Sim(3) pose-graph
optimization" until source evidence shows a fuller LBA formulation.

`Sigma_ij` remains an explicit interface combining descriptor consistency and
3D alignment residual evidence.

## 15. Versioning And Invalidation

Every DEM, descriptor, index, match, and edge carries a signature containing:

- Scale status and scale version.
- Canonical frame version.
- Lattice origin.
- `mpp`.
- Tile size and `tile_px`.
- Reducer and normalization version.
- DINO weight hash, layer, facet, input size, and context.
- Profile and code version.

Gallery and query signatures must match. Otherwise retrieval fails with a
clear rebuild requirement.

Backend corrections can invalidate:

- Point positions.
- Tile occupancy.
- DEM values.
- Normalization statistics.
- DINO descriptors.
- FAISS index entries.
- Geometric loop measurements.

Offline M0-M4 uses deterministic full rebuilds first. Incremental invalidation
is added only after correctness is established.

## 16. Error Handling

Algorithms return structured results or structured rejection reasons.

Examples:

- Insufficient valid depth.
- Insufficient plane inliers.
- Ambiguous canonical axes.
- Invalid or missing metric anchor.
- Empty DEM tile.
- Descriptor signature mismatch.
- HNSW index version mismatch.
- Too few VPR matches.
- Degenerate Sim(3) geometry.
- Optimizer non-convergence.

No stage silently substitutes units, coordinate frames, profiles, weights, or
devices.

## 17. Testing

### 17.1 Synthetic Geometry

- Plane recovery with noise and outliers.
- Plane normal and PCA axis sign stability.
- Metric and relative scale propagation.
- Stable tile IDs during map expansion.
- Pixel and tile boundary behavior.
- Mean, max, and numerically stable softmax reducers.
- Empty pixels and masks.

### 17.2 Real Submap Review

- Exported TUM submap metadata.
- Raw and filtered point clouds.
- Plane inlier visualization.
- Height histograms.
- Grayscale DEM tiles.
- Human color visualization.
- Profile and scale annotations.

### 17.3 Embedding And Retrieval

- Output dimension and L2 normalization.
- Deterministic CPU inference tolerance.
- Query/gallery signature rejection.
- Exact retrieval ground truth.
- HNSW recall@k and latency.
- Raw tile matches and parent-submap votes.

### 17.4 Sim(3)

- Composition and inverse.
- Exp/Log round trip.
- Jacobian checks where supported.
- Synthetic relative-pose graph recovery.
- Gauge anchoring.
- Outlier and degeneracy rejection.
- Optimization before/after error.

### 17.5 Review Gates

Each milestone delivers:

- Source code with algorithm-oriented comments.
- Formula-to-code mapping.
- Test output.
- Reviewable artifacts.
- Known deviations and evidence labels.

Implementation does not proceed to the next milestone until the user reviews
the current milestone.

## 18. Commenting Standard

Comments at critical algorithms explain:

- Coordinate frame and transform direction.
- Units and scale status.
- Array or tensor shapes.
- Formula and paper section.
- Parameter evidence.
- Numerical stability measures.
- Rejection or invalidation behavior.

Comments should not narrate obvious assignments or loops.

## 19. External Source Management

`external_sources/` is ignored except for tracked metadata:

- `manifest.yaml`
- `README.md`
- License notes
- Commit hashes
- Migration notes and source-to-destination mapping

Expected investigations include:

- AnyLoc.
- DINOv2 reference feature extraction.
- Reliable Sim(3) optimization.
- Any source that resolves DEM weighting or geometric edge construction.

No algorithm is copied without provenance and license review.

## 20. Git And Large Files

Track:

- Baseline core source.
- `vggt_slam_pp/`.
- Configurations.
- Environment definitions.
- Tests.
- Documentation.
- External source manifests.
- Small deterministic fixtures.

Ignore:

- `.DS_Store`
- `.superpowers/`
- `VGGT-SLAM-version1.0.zip`
- Extracted demo data when redundant.
- `weights/`
- `data/`
- `artifacts/`
- Downloaded third-party repositories.
- Build, environment, cache, log, and visualization output.

Large files retain fixed expected paths and SHA-256 metadata. Preparation
scripts verify presence and hashes on AutoDL rather than downloading
implicitly during algorithm execution.

Known hashes:

```text
VGGT-SLAM-version1.0.zip
3283a68428ce86f95313e94d6a7a22b0b5f150c632c90aa57675ba35d5132aee

weights/dinov2_vitb14_pretrain.pth
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73

weights/dino_salad.ckpt
6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada
```

## 21. Deferred Decisions

These decisions are intentionally deferred to milestone-specific validation:

- Legal metric scale source for each dataset.
- TUM and KITTI depth thresholds.
- Plane RANSAC thresholds.
- `mpp`, tile size, and DINO input size per profile.
- Interpretation of 90k pixels and 4096 tiles.
- DINO layer, facet, Gaussian sigma, and visibility mask.
- HNSW construction and search parameters.
- AnyLoc VLAD configuration.
- Relative Sim(3) measurement algorithm.
- Edge information matrix.
- Local graph window and scheduler frequency.

Each deferred item has an interface and evidence label, so it can be resolved
without restructuring the system.

## 22. First Implementation Scope

After this design is approved, the first implementation plan covers only:

1. M0 baseline provenance and AutoDL runbook.
2. Top-level Git ignore rules.
3. External source manifest structure.
4. Reviewable submap cache schema and exporter.
5. M1 synthetic tests and standalone DEM implementation.
6. TUM `freiburg1_desk` DEM review artifacts.

M2-M4 remain interface scaffolding and documented research tasks until M1 is
reviewed.
