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
import numpy as np
import pymupdf 

def pdf_to_pil(pdf_path, page_number=0):
    # Open the PDF file.
    doc = pymupdf.open(pdf_path)
    # Load the specified page.
    page = doc.load_page(page_number)
    # Render the page to a pixmap.
    pix = page.get_pixmap()
    # Attempt to get a PIL image directly (available in recent versions of PyMuPDF).
    try:
        img = pix.get_pil_image()
    except AttributeError:
        # Fallback: create a PIL image manually if get_pil_image() isn't available.
        from PIL import Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img

def process_directory(dir_path, max_epochs=60, generate_plot=False):
    """
    For a given directory (an immediate subdirectory of the base),
    check if it contains .npz files (only in that directory, not nested).
    Then, extract the epoch numbers from filenames (pattern: 'ep<number>.npz').
    Reject the directory if the maximum epoch is less than max_epochs.
    Otherwise, run the evaluation script with fixed parameters (ipe=1, n_steps=5000)
    and return (dir_path, plot_file, max_epoch) if the plot is created.
    """
    # Look for .npz files only in this directory.
    # skip if train is in the path

    plot_file = os.path.join(dir_path, "Full_learning_dynamics_multi-class_.pdf")
    if os.path.isfile(plot_file) and not generate_plot:
        print(f"Plot file already exists in {dir_path}: {plot_file}")
        return (dir_path, plot_file, None)
    
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

    
    batch_size = 256
    ipe = 5000 // batch_size # corresponds to batch_size=256
    # check if the current directory last part is "val"
    if os.path.basename(dir_path) == "val":
        # check if the parent parent of this directory has config.yaml
        parent_dir = os.path.dirname(os.path.dirname(dir_path))
        config_file = os.path.join(parent_dir, "config.yaml")
        if not os.path.isfile(config_file):
            print(f"we are in a maskgit data, but the config.yaml file is not found in {parent_dir}")
            return None
        else:
            #load yaml and get batch size 
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                #batch_size=$(python -c "import yaml, sys; cfg = yaml.safe_load(open(sys.argv[1])); print(cfg['data']['params']['batch_size'])" "$yaml_file")
                batch_size = config['data']['params']['batch_size']
                print(f"Batch size: {batch_size}")
                ipe = 5000 // batch_size
        

    print(f"Directory: {dir_path}")
    print(f"  Max epoch found: {max_epoch}")
    print(f"  Using fixed parameters: ipe = {ipe}, batch_size = {batch_size}")

    # Run the evaluation script.
    cmd = [
        "python", "plotting.py",
        "--exp_dir", dir_path + "/",
        "--scale_factor", str(ipe),
    ]
    print("Running command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script for {dir_path}: {e}")
        return None

    # Check if the evaluation script created the plot file.
    plot_file = os.path.join(dir_path, "Full_learning_dynamics_multi-class_.pdf")
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
        if "single-body_2d_3classes" not in parts:
            #then it is a maskgit data where the label is the name of the parent of the parent of the current directory
            parent_parent = os.path.dirname(os.path.dirname(dir_path))
            label = os.path.basename(parent_parent)
        elif "celeba-3classes-smiling-10000_100" in parts:
            idx = parts.index("celeba-3classes-smiling-10000_100")
            # Take the next three parts; if there are fewer than three, take whatever is available.
            label_parts = parts[idx+1 : idx+5]
            label = "/".join(label_parts)
        else: 
            idx = parts.index("single-body_2d_3classes")
            # Take the next three parts; if there are fewer than three, take whatever is available.
            label_parts = parts[idx+1 : idx+5]
            label = "/".join(label_parts)
    except ValueError:
        # If "single-body_2d_3classes" is not found, fall back to the base directory name.
        label = os.path.basename(norm_path)
    return label.replace("_", " ").replace("/", " ")

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
            #check if png or pdf
            if plot_file.endswith(".pdf"):
                img = pdf_to_pil(plot_file)
                img = img.convert("RGB")  # Convert to RGB if needed
                img = np.array(img)
            elif plot_file.endswith(".png"):
                img = mpimg.imread(plot_file)
        except Exception as e:
            print(f"Error reading image {plot_file}: {e}")
        

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
    parser.add_argument("base_dir", help="Path to the base directory to scan for .npz files.")
    parser.add_argument("--max_epochs", type=int, default=60,
                        help="Minimum max epoch required in a directory for processing (default: 60).")
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
        #print(f"Checking directory: {root}")
        if glob.glob(os.path.join(root, '*.npz')):
            if "/train" in root and "maskgit" in root:
                # Skip if "train" is in the path for maskgit data.
                #print(f"Skipping {root}: contains 'train' in the path for maskgit data.")
                continue
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
