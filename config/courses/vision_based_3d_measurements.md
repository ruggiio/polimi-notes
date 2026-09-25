# Scheda corso — VISION BASED 3D MEASUREMENTS

Misure 3D basate su visione, Politecnico di Milano, A.A. 2026/27.

## Glossario

- pinhole camera model, intrinsic parameters, extrinsic parameters, focal length, principal point, lens distortion
- camera calibration, checkerboard, reprojection error, homography, epipolar geometry, fundamental matrix, essential matrix
- stereo vision, disparity, triangulation, rectification, baseline, depth map, point cloud
- structured light, laser triangulation, time of flight, ToF, photogrammetry, structure from motion, SfM, bundle adjustment
- digital image correlation, DIC, feature detection, SIFT, ORB, RANSAC, sub-pixel accuracy, measurement uncertainty

## Notazione

- Coordinate immagine $(u, v)$, coordinate mondo $(X, Y, Z)$, matrice intrinseca $K$, rotazione $R$, traslazione $\mathbf{t}$
- Proiezione $\tilde{\mathbf{x}} = K [R \mid \mathbf{t}] \tilde{\mathbf{X}}$

## Stile del docente

- Le derivazioni geometriche vanno in prosa con `align`; le procedure di calibrazione come elenchi numerati
