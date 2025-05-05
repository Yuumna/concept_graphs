#!/usr/bin/env python3
import os
import sys
import glob
import re
import subprocess
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def process_directory(dir_path, max_epochs=1000):
    """
    For a given directory (an immediate subdirectory of the base),
    check if it contains .npz files (only in that directory, not nested).
    Then, extract the epoch numbers from filenames (pattern: 'ep<number>.npz').
    Reject the directory if the maximum epoch is less than max_epochs.
    Otherwise, run the evaluation script with fixed parameters (ipe=1, n_steps=5000)
    and return (dir_path, plot_file, max_epoch) if the plot is created.
    """
    # Look for .npz files only in this directory.
    npz_files = glob.glob(os.path.join(dir_path, "*.npz"))
    if not npz_files:
        return None

    epoch_numbers = []
    # Process each .npz file to extract epoch numbers.
    for file_path in npz_files:
        basename = os.path.basename(file_path)
        # Expecting filename format like "image_111_ep5499.npz"
        match = re.search(r"ep(\d+)\.npz$", basename)
        if match:
            try:
                epoch_num = int(match.group(1))
                epoch_numbers.append(epoch_num)
            except ValueError:
                continue

    if not epoch_numbers:
        print(f"No epoch numbers found in npz files in {dir_path}")
        return None

    max_epoch = max(epoch_numbers)
    if max_epoch < max_epochs:
        print(f"Skipping {dir_path}: max epoch {max_epoch} < {max_epochs}")
        return None

    # Fixed parameters.
    ipe = 1
    n_steps = 5000
    print(f"Directory: {dir_path}")
    print(f"  Max epoch found: {max_epoch}")
    print(f"  Using fixed parameters: ipe = {ipe}, n_steps = {n_steps}")

    # Run the evaluation script.
    cmd = [
        "python", "probes/evaluate_and_plot.py",
        "--path_images", dir_path,
        "--ipe", str(ipe),
        "--n_steps", str(n_steps)
    ]
    print("Running command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script for {dir_path}: {e}")
        return None

    # Check if the evaluation script created the plot file.
    plot_file = os.path.join(dir_path, "plot_probes.png")
    if os.path.isfile(plot_file):
        return (dir_path, plot_file, max_epoch)
    else:
        print(f"Plot file not found in {dir_path}")
        return None

def extract_label(dir_path):
    """
    Extract a label from the full path of the directory that contains the .npz files.
    The function finds the folder "single-body_2d_3classes" in the path, and then
    returns the next three path components (joined with "/") as the label.
    For example, given a path like:
    
    /.../single-body_2d_3classes/U-Net/pixel/discrete/sec_run/...
    
    the extracted label will be "U-Net/pixel/discrete".
    """
    norm_path = os.path.normpath(dir_path)
    parts = norm_path.split(os.sep)
    try:
        idx = parts.index("single-body_2d_3classes")
        # Take the next three parts; if there are fewer than three, take whatever is available.
        label_parts = parts[idx+1 : idx+4]
        label = "/".join(label_parts)
    except ValueError:
        # If "single-body_2d_3classes" is not found, fall back to the base directory name.
        label = os.path.basename(norm_path)
    return label

def aggregate_plots(plot_infos, output_file="aggregated_plots.png"):
    """
    Given a list of (dir_path, plot_file, max_epoch), aggregate all plots into a grid.
    Each subplot is labeled with the extracted folder name (the one immediately after
    'single-body_2d_3classes').
    """
    if not plot_infos:
        print("No valid plots to aggregate.")
        return

    n_images = len(plot_infos)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)
    print(f"Aggregating {n_images} plots into a grid of {n_rows} rows x {n_cols} columns.")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Flatten the axes array for easy iteration.
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for i, (dir_path, plot_file, max_epoch) in enumerate(plot_infos):
        try:
            img = mpimg.imread(plot_file)
        except Exception as e:
            print(f"Error reading image {plot_file}: {e}")
            continue

        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        # Use the extracted label for the title.
        title = extract_label(dir_path)
        ax.set_title(title, fontsize=10)

    # Turn off any extra subplots.
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Aggregated plot saved as {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Process directories containing .npz files and aggregate generated plots."
    )
    parser.add_argument("--base_dir", help="Path to the base directory to scan for .npz files.")
    parser.add_argument("--max_epochs", type=int, default=1000,
                        help="Minimum max epoch required in a directory for processing (default: 1000).")
    parser.add_argument("--output_file", default="aggregated_plots.png",
                        help="Filename for the aggregated plot output (default: aggregated_plots.png).")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        sys.exit(1)

    subdirs = []
    for root, dirs, files in os.walk(base_dir):
        # Check if any .npz file is present in the current directory.
        print(f"Checking directory: {root}")
        if glob.glob(os.path.join(root, '*.npz')):
            subdirs.append(root)

    if not subdirs:
        print("No subdirectories with .npz files found in", base_dir)
        sys.exit(0)

    plot_infos = []
    # Process each subdirectory that contains .npz files.
    for subdir in subdirs:
        result = process_directory(subdir, max_epochs=args.max_epochs)
        if result:
            plot_infos.append(result)

    # Aggregate plots from all valid directories.
    aggregate_plots(plot_infos, output_file=args.output_file)

if __name__ == "__main__":
    main()
