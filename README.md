# Classical Relative Pose Estimation Pipeline

A C++ implementation of a classical relative pose estimation pipeline : SIFT correspondence extraction, RANSAC / AC-RANSAC outlier rejection, Kanatani's FNS refinement, and essential matrix decomposition with cheirality-based disambiguation. This project was developed during a 6-month internship at the Imagine Laboratory, École des Ponts ParisTech (ENPC), in partial fulfillment of my masters degree at Sorbonne Université.

**Author:** Nesrine Naaman
**Supervisor:** Prof. Pascal Monasse
**University Tutors:** Isabelle Bloch, Dominique Béréziat

## Project Structure

```
.
├── src/                    # Pipeline source code
├── external/
│   ├── libOrsa/            # RANSAC / AC-RANSAC / linear algebra library
│   │                       
│   ├── Imagine/             # SIFT feature extraction
│   └── CppUnitLite/          # Unit testing utilities
├── data/
│   └── config/
│       └── dataset.txt      # Dataset path configuration
├── tests/
│   ├── synthetic/            # Noiseless synthetic ground-truth pipeline validation
│   └── noise_injection/       # Synthetic tests with injected correspondence noise,
│                              # comparing all four of Kanatani's
│                              # estimators (FNS, modified FNS, HEIV, renormalization)
├── results/
│   ├── classical_orsa/         # TUM RGB-D results using the AC-RANSAC backend
│   ├── classical_ransac/       # TUM RGB-D results using the original RANSAC backend
│   └── distorted/              # Results computed on raw, distortion-uncorrected images
├── CMakeLists.txt
└── README.md
```

## Dependencies

- CMake ≥ 3.10
- Eigen3
- GSL
- Imagine++
- A C++17-compatible compiler

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Running the Pipeline

The executable accepts up to four optional command-line arguments:

```bash
./Relative_Pose_Estimator [skipping_frame] [method] [stride] [pair_index]
```

| Argument         | Description | Default |
|------------------|-------------|---------|
| `skipping_frame` | Frame gap between the two images forming a pair (e.g., `2`, `5`, `10`, `12`) | `5` |
| `method`         | Fundamental matrix refinement method: `1` = Kanatani's FNS, `2` = Gauss-Newton | `1` (FNS) |
| `stride`         | Step size used when sampling starting indices from the TUM association sequence | `5` |
| `pair_index`     | (Optional) If provided, runs the pipeline on a single pair, identified by its index into the TUM associations file, and prints detailed per-pair diagnostics instead of running the full batch evaluation |  |

**Example — batch evaluation** (skip = 10, FNS, stride = 5; evaluates all valid pairs and writes CSV/summary output):
```bash
./Relative_Pose_Estimator 10 1 5
```

**Example — single-pair inspection** (skip = 10, FNS, stride = 5, inspecting the pair starting at association index 900):
```bash
./Relative_Pose_Estimator 10 1 5 900
```

Each batch run produces two output files:
- `pose_results_<method>_skip<N>.csv` : per-pair results, including image filenames, ground-truth rotation and translation magnitude, inlier count, match coverage, and rotation/translation error.
- `summary_results_<method>_skip<N>.txt` : aggregate mean/median statistics for that configuration, along with the number of pairs evaluated and discarded.

## Reproducibility

Random sampling (used internally by the AC-RANSAC backend) is seeded with
a fixed value (`srand(42)`) at program start, ensuring that batch and
single-pair runs on the same input data produce identical, reproducible
results.

## Dataset Configuration

The dataset path is set in `data/config/dataset.txt`, and should be updated to point to wherever the TUM RGB-D dataset has been downloaded locally.

In this project's own development, the dataset was stored locally on disk. A copy of the dataset sequence used, along with all generated result files, is also available via the shared OneDrive link in the "Data and Results Availability" section below.

## Results

Precomputed results on the TUM RGB-D `fr1/room` sequence are provided in
the `results/` directory:

- **`results/classical_orsa/`**: results using the AC-RANSAC (ORSA) backend, across all tested frame-skip values.
- **`results/classical_ransac/`**: results using the original, fixed-threshold RANSAC backend, for comparison against AC-RANSAC.
- **`results/distorted/`**: results computed on raw, distortion-uncorrected images, for comparison against the fully distortion-corrected results used elsewhere in this work.

## Data and Results Availability

The SRPose inference implementation, the `fr1/room` sequence, generated
result files for both methods, and the scripts used to produce the plots
and figures in this work are available via the following shared GoogleDrive
folder:

https://shorturl.at/ggXZ5

## Attribution

The RANSAC, AC-RANSAC, and linear algebra (`libOrsa` / `libNumerics`)
implementations used in this project are from Professor Pascal Monasse obtained from the following IPOL article:

> Lionel Moisan, Pierre Moulon, and Pascal Monasse. Fundamental Matrix of a Stereo Pair, with A Contrario Elimination of Outliers. Image Processing On Line, 6:89–113, 2016.
> https://doi.org/10.5201/ipol.2016.147

SRPose evaluation used the official implementation released by its
authors: https://github.com/frickyinn/SRPose

TUM RGB-D dataset, ground truth, and the `associate.py` script are from:
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/

See the accompanying report for complete methodology, results, and
discussion.
