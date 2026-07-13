import os
import laspy
import torch

from flooder import (
    generate_noisy_torus_points_3d,
    flood_complex, 
    generate_landmarks)

DEVICE = "cuda"
n_pts = 1_000_000  # Number of points to sample from torus
n_lms = 1_000      # Number of landmarks for Flood complex

# las = laspy.read(os.path.join(os.path.dirname(__file__), "data", "dales-1m.las"))

# pts = torch.tensor(las.xyz, device=DEVICE)

pts = generate_noisy_torus_points_3d(n_pts).to(DEVICE)
lms = generate_landmarks(pts, n_lms)

stree = flood_complex(pts, lms, return_simplex_tree=True)
stree.compute_persistence()
ph = [stree.persistence_intervals_in_dimension(i) for i in range(3)]



