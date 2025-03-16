from linear_classifier_3classes import evaluate, MLP, calculate_accuracy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
import json

#path_images = 'output/single-body_2d_3classes/latent_None/H32-train1/06-03-16-37_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1'
# cont latent 
path_images= 'output/single-body_2d_3classes/latent_None/H32-train1/06-03-16-38_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1'
model_path = '/work/dlclarge2/aliy-maskgit/concept_graphs/probes/linear-classifier_single-body_2d_3classes_multi-class.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = [f for f in os.listdir(path) if f.startswith('image_') and f.endswith('.npz') ]#and int(f.split('_ep')[-1].split('.')[0]) > 5000]

    def __len__(self):
        return len(self.files) 

    def __getitem__(self, idx):
        file = self.files[idx]
        ep = int(file.split('_ep')[-1].split('.')[0])
        data = np.load(os.path.join(self.path, file), allow_pickle=True)
        image = data['x_gen'].astype(np.float32)
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

def evaluate(model, iterator, criterion, device):

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
            print(f"y before shape: {y[0].shape} , and y: {y[0]}")
            print(f"y_pred[0] shape: {y_pred[0].shape}")
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
                "step": int(ep.item()),  # Convert tensor to integer
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


dataset = CustomDataset(path_images)
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

criterion = nn.CrossEntropyLoss()
if __name__ == "__main__":
    test_loss, test_acc = evaluate(model, dataloader, criterion, device)
    # save the results
    print(f'\tTest Loss: {test_loss:.3f} | test Acc: {test_acc[0]*100:.2f}% {test_acc[1]*100:.2f}% {test_acc[2]*100:.2f}%')
    with open(os.path.join(path_images, 'test_acc.txt'), 'w') as f:
        f.write(f'\tTest Loss: {test_loss:.3f} | test Acc: {test_acc[0]*100:.2f}% {test_acc[1]*100:.2f}% {test_acc[2]*100:.2f}%')

