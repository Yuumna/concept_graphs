#!/bin/bash

# List of exp_dir paths you want to evaluate
paths=(
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/DiT/latent/continuous/sec_run/H32-train1/30-03-20-17_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/DiT/latent/discrete/sec_run/H32-train1/30-03-20-17_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/DiT/pixel/continuous/first_run/H32-train1/29-03-22-47_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/DiT/pixel/discrete/first_run/H32-train1/29-03-22-47_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/U-Net/latent/continuous/first_run/H32-train1/29-03-22-41_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/U-Net/latent/discrete/first_run/H32-train1/29-03-22-41_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/U-Net/pixel/continuous/sec_run/H32-train1/30-03-22-30_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
    "/work/dlclarge2/aliy-maskgit/concept_graphs/results/output_all/single-body_2d_3classes/U-Net/pixel/discrete/sec_run/H32-train1/30-03-22-30_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1"
)

# Loop over each path and run the Python script
for path in "${paths[@]}"; do
    echo "Evaluating: $path"
    python probes/evaluate.py --path_images "$path"
done
