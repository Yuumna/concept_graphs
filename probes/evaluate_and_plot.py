from linear_classifier_3classes import evaluate, MLP, calculate_accuracy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
import json
import argparse

import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

line_styles = {
    "000": ("#22a6ff", "solid"),
    "001": ("#22a6ff", "dashed"),
    "010": ("#22a6ff", "dashdot"),
    "100": ("#22a6ff", "dotted"),
    "011": ("#ff95ca", "solid"),
    "101": ("#ff95ca", "dashed"),
    "110": ("#ff95ca", "dashdot"),
    "111": ("#ff00ae", "solid"),
}


def get_dict(path):
    
    with open(path, "r") as f:
        log_data = json.load(f)

    data_by_concept = {}

    for entry in log_data:
        step = entry["step"]
        concept_code = entry["concept_code"]
        accuracy = entry["accuracy"] * 100

        if concept_code not in data_by_concept:
            data_by_concept[concept_code] = {"steps": [], "accuracies": []}

        data_by_concept[concept_code]["steps"].append(step)
        data_by_concept[concept_code]["accuracies"].append(accuracy)
    sorted_keys = sorted(data_by_concept.keys())
    for concept_code in sorted_keys:
        steps = np.array(data_by_concept[concept_code]["steps"])
        accuracies = np.array(data_by_concept[concept_code]["accuracies"])

        sorted_indices = np.argsort(steps)
        data_by_concept[concept_code]["steps"] = steps[sorted_indices]
        data_by_concept[concept_code]["accuracies"] = accuracies[sorted_indices]
    return data_by_concept


class CustomDataset(Dataset):
    def __init__(self, path, n_samples=1000): #=4):
        self.n_samples = n_samples
        self.path = path
        self.files = [f for f in os.listdir(path) if f.startswith('image_') and f.endswith('.npz') and int(f.split('_ep')[-1].split('.')[0]) < self.n_samples]
        #get the first n_samples files
        #self.files = self.files[:self.n_samples]

    def __len__(self):
        return len(self.files) 

    def __getitem__(self, idx):
        file = self.files[idx]
        ep = int(file.split('_ep')[-1].split('.')[0])
        data = np.load(os.path.join(self.path, file), allow_pickle=True)
        print(f"Loading file: {file}")
        print(f"Loaded data keys: {data.keys()}")

        if 'x_gen' in data:
            image = data['x_gen'].astype(np.float32)
        else:
            # Load the first available key as a fallback
            first_key = list(data.keys())[0]
            print(f"'x_gen' not found in file: {file}. Using the first available key: {first_key}")
            image = data[first_key].astype(np.float32)
        #image = data['x_gen'].astype(np.float32)
        label_str = file.split('_')[1]
        label = np.array([int(digit) for digit in label_str], dtype=np.int64)
        label = np.tile(label, (image.shape[0], 1))
        labels = {
            0: torch.tensor(label[:, 0], dtype=torch.long),
            1: torch.tensor(label[:, 1], dtype=torch.long),
            2: torch.tensor(label[:, 2], dtype=torch.long)
        }
        #print(f"Loaded image shape: {image.shape}, label:{labels}")
        return torch.tensor(image, dtype=torch.float32), labels, ep, label_str

def evaluate(model, iterator, criterion, device, ipe=1):
    """
    Evaluate the model on the validation set.
    """

    epoch_loss = 0
    #epoch_acc = 0
    epoch_acc = {0: 0, 1: 0, 2: 0}
    log_data = []


    model.eval()
    with torch.no_grad():
        for (x, y, ep, label_str) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x[:50].to(device)
            #y = _y[key].to(device)
            x= x.squeeze(0)
            y = [y[key][:50].to(device).transpose(1,0).squeeze(1) for key in y.keys()]
            y_pred = model(x)
            
            #print(f"y shape: {y.shape}, unique values: {torch.unique(y)}")
            #print(f"y before shape: {y[0].shape} , and y: {y[0]}")
            #print(f"y_pred[0] shape: {y_pred[0].shape}")
            # transpose y to match the shape of y_pred
            #print(f"y before shape: {y[0].shape}")

            loss = criterion(y_pred[0], y[0]) + criterion(y_pred[1], y[1]) + criterion(y_pred[2], y[2])
            acc = {}
            acc[0] = calculate_accuracy(y_pred[0], y[0])
            acc[1] = calculate_accuracy(y_pred[1], y[1])
            acc[2] = calculate_accuracy(y_pred[2], y[2])
            epoch_loss += loss.item()
            #epoch_acc += acc.item()
            epoch_acc[0] += acc[0].item()
            epoch_acc[1] += acc[1].item()
            epoch_acc[2] += acc[2].item()
            
            acc_compositional = acc[0] * acc[1] * acc[2]
            print(f"acc_compositional: {acc_compositional}")
            print(f"acc[0]: {acc[0]}")
            print(f"acc[1]: {acc[1]}")
            print(f"acc[2]: {acc[2]}")
            print(f"step: {ep}")
            print(f"label_str: {label_str}, {label_str[0]}")
            # create json file with the logs of accuracy and ep for each class
            log_data.append({
                "step": int(ep.item()) * ipe,
                "concept_code": str(label_str[0]),
                "accuracy": acc_compositional.item()  # Single compositional accuracy value
            })
            
    # Save Logs to JSON File
    json_path = os.path.join(path_images, "steps_acc.json")
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"epoch_acc: {epoch_acc}")
    epoch_acc[0] /= len(iterator) 
    epoch_acc[1] /= len(iterator)
    epoch_acc[2] /= len(iterator)
    
    print(f"len(iterator): {len(iterator)}")
    return epoch_loss / len(iterator), epoch_acc #/ len(iterator)


def plot(data, line_styles, save_path):
    plt.figure(figsize=(7, 5))

    for concept_code in line_styles.keys():
        steps = np.array(data[concept_code]["steps"])
        accuracies = np.array(data[concept_code]["accuracies"])

        color, linestyle = line_styles.get(concept_code, ("gray", "solid"))
        plt.plot(steps, accuracies, linestyle=linestyle, color=color, label=concept_code)

    plt.xscale("log")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Optimization Steps")

    plt.legend(title="Concept Code", fontsize=9)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")



criterion = nn.CrossEntropyLoss()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--path_images', type=str, required=True, help='Path to the images directory')
    parser.add_argument('--model_path', type=str, default='/work/dlclarge2/aliy-maskgit/concept_graphs/probes/linear-classifier_single-body_2d_3classes_multi-class.pt', help='Path to the model file')
    parser.add_argument('--n_steps', type=int, default=10000, help='Number of epochs to plot probes')
    parser.add_argument('--ipe', type=int, default=1, help='Number of iterations per epoch')
    args = parser.parse_args()
    path_images = args.path_images
    model_path = args.model_path
    n_epochs = args.n_steps / args.ipe
    ipe = args.ipe


    dataset = CustomDataset(path_images, n_samples=n_epochs)
    dataloader = DataLoader(dataset, shuffle=True)

    dataset = "single-body_2d_3classes"
    properties_json = f"properties_{dataset}.json"
    with open(properties_json, 'r') as f:
        properties = json.load(f)

    pixel_size = 28 
    INPUT_DIM = pixel_size * pixel_size * 3

    OUTPUT_DIMS = [len(properties[key]) for key in ["shapes", "colors", "sizes"]]
    n_class_color = 2  #
    OUTPUT_DIMS[1] = n_class_color

    model = MLP(INPUT_DIM, OUTPUT_DIMS).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    
    test_loss, test_acc = evaluate(model, dataloader, criterion, device, ipe=ipe)
    # save the results
    print(f'\tTest Loss: {test_loss:.3f} | test Acc: {test_acc[0]*100:.2f}% {test_acc[1]*100:.2f}% {test_acc[2]*100:.2f}%')
    with open(os.path.join(path_images, 'test_acc.txt'), 'w') as f:
        f.write(f'\tTest Loss: {test_loss:.3f} | test Acc: {test_acc[0]*100:.2f}% {test_acc[1]*100:.2f}% {test_acc[2]*100:.2f}%')

    # create plot
    # Load the JSON data
    json_path = os.path.join(path_images, "steps_acc.json")
    data = get_dict(json_path)
    # Plot the data
    plot(data, line_styles, os.path.join(path_images, "plot_probes.png"))
    print(f"Plot saved to {os.path.join(path_images, 'plot_probes.png')}")

