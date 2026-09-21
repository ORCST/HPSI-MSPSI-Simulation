# HPSI and MSPSI Simulation

This program provides a simple simulation of HPSI and MSPSI associated with our paper:

Both implementations follow the core reconstruction principles of their respective methods. They use the same simulated scene, calibration parameters, camera ROI, and triangulation routine, and both employ CUDA acceleration. Implementation details and optimization choices may affect runtime.

## Simulation Settings

- **HPSI frequencies:** 1, 16, 32, 40, 48
- **MSPSI frequencies:** 8, 16, 24, 32, 40
- **Camera ROI:** 542 × 369 pixels, totaling 199,998 reconstructed points

## Running the Program

Place `psi.exe` and `scene.bin` in the same folder and launch `psi.exe`. Alternatively, compile the program from source using the supplied `CMakeLists.txt`.

| Control | Action |
|---------|--------|
| **H** | Switch to HPSI |
| **M** | Switch to MSPSI |
| **N** | Enable or disable simulated noise (0.4 DN RMS when enabled) |
| **Left mouse button + drag** | Rotate the view |
| **Mouse wheel** | Zoom |
| **Esc** or close the viewer | Stop the program |

## Visualization and Timing

The left panel displays the reference point cloud, while the right panel displays the reconstructed point cloud. Both panels use the same Jet depth-color mapping.

Per-frame computation times are printed to the console in milliseconds. Reconstruction timing excludes scene generation and screen rendering. The first three frames of each method are marked as warm-up frames.

## Requirements

- Windows
- A supported NVIDIA GPU with a compatible driver

Running the supplied executable does not require a separate installation of the CUDA Toolkit.

Building from source requires CMake, the CUDA Toolkit, and a compatible C++ compiler. Runtime and numerical results may vary across computers.

## Example Runtime

The table below shows individual frame timings from an example run on an NVIDIA GeForce RTX 4060 GPU, reconstructing 199,998 points without simulated noise. These are single-frame measurements rather than averages.

| Stage | HPSI | MSPSI |
|-------|---------------------:|----------------------:|
| Input upload | 1.007 | 1.966 |
| LTC computation, including coefficients | 4.745 | 7.145 |
| Peak search | 9.022 | 2960.549 |
| Final subpixel refinement | 0.142 | 0.284 |
| Triangulation | 0.021 | 0.020 |
| Result download | 0.318 | 0.361 |
| **Total reconstruction** | **15.276** | **2970.370** |

Total reconstruction corresponds to the program's `total` field and excludes scene generation and screen rendering. Stage timings use CUDA events, whereas the total uses wall-clock timing, so their sum may differ slightly from the reported total. The MSPSI frame shown here was marked as warm-up. Actual runtime may vary with hardware and execution conditions.

## Notes

The simulation is intended for method demonstration and reproducibility. The measured runtime ratio should not be interpreted as an intrinsic performance limit of either method.
