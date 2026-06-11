# VGGT-SLAM 1.0 Baseline Provenance

- Upstream: https://github.com/MIT-SPARK/VGGT-SLAM
- Local archive: `VGGT-SLAM-version1.0.zip`
- Archive SHA-256: `f34897e5745c6380dfd819bf87c8a016aebb8e9ffe7a0025304015fa7b0f0411`
- Canonical source directory: `VGGT-SLAM-version1.0/`
- Replaced modified copy: `/tmp/vggt-slam-modified-backup-20260611`
- Replaced clean duplicate: `/tmp/vggt-slam-clean-duplicate-20260611`

The default path is 15DoF SL(4). `--use_sim3` estimates scale outside the
graph and optimizes 6DoF Pose3 factors; this project names it
`baseline_sim3_compat`.

The canonical directory is extracted directly from the verified archive.
Local model weights, datasets, demo frames, and generated artifacts are
provisioned separately and are excluded from Git.
