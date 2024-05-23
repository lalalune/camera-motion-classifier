import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from enum import Enum
import argparse

# Enum for camera motion labels
class CameraMotion(Enum):
    STATIC = 0
    ORBIT_LEFT = 1
    ORBIT_RIGHT = 2
    ORBIT_UP = 3
    ORBIT_DOWN = 4
    PAN_LEFT = 5
    PAN_RIGHT = 6
    CRANE_UP = 7
    CRANE_DOWN = 8
    TILT_LEFT = 9
    TILT_RIGHT = 10
    TILT_UP = 11
    TILT_DOWN = 12
    ZOOM_IN = 13
    ZOOM_OUT = 14
    DOLLY_IN = 15
    DOLLY_OUT = 16

# Dataset class for actual video files
class VideoDataset(Dataset):
    def __init__(self, video_folder, label_file):
        self.video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]
        self.label_map = {motion.name: motion.value for motion in CameraMotion}
        self.labels = self.load_labels(label_file)

    def load_labels(self, label_file):
        labels = {}
        with open(label_file, "r") as f:
            for line in f:
                video_name, label_name = line.strip().split()
                labels[video_name] = self.label_map[label_name]
        return labels

    def __len__(self):
        return len(self.video_files) * 3  # 3 examples per video

    def __getitem__(self, idx):
        video_idx = idx // 3
        example_idx = idx % 3

        video_path = self.video_files[video_idx]
        video_name = os.path.basename(video_path)
        label = self.labels[video_name]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if example_idx == 0:
            frame_indices = [0, 1]
        elif example_idx == 1:
            frame_indices = [total_frames // 2, total_frames // 2 + 1]
        else:
            frame_indices = [total_frames - 2, total_frames - 1]

        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))  # Change input size to 128x128
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()
        
        if len(frames) < 2:
            raise ValueError(f"Video {video_path} has less than 2 frames.")
        
        flow = cv2.calcOpticalFlowFarneback(frames[0], frames[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.transpose(flow, (2, 0, 1))
        flow = torch.from_numpy(flow).float()
   
        return flow, label

# Neural network model
class CameraMotionClassifier(nn.Module):
    def __init__(self):
        super(CameraMotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Update input size for fc1
        self.fc2 = nn.Linear(512, len(CameraMotion))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for flows, labels in dataloader:
        flows = flows.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(flows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for flows, labels in dataloader:
            flows = flows.to(device)
            labels = labels.to(device)
            
            outputs = model(flows)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return epoch_loss, accuracy, predictions, ground_truths

def predict_video(model, video_path, device, num_samples=5):
    model.eval()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    predictions = []
    for i in range(num_samples):
        frame_idx = i * (total_frames // (num_samples + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        frames = []
        for _ in range(2):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        if len(frames) < 2:
            continue

        flow = cv2.calcOpticalFlowFarneback(frames[0], frames[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.transpose(flow, (2, 0, 1))
        flow = torch.from_numpy(flow).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(flow)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    cap.release()

    if not predictions:
        raise ValueError(f"No valid frames found in the video {video_path}.")

    # Find the most common prediction (majority consensus)
    consensus_prediction = max(set(predictions), key=predictions.count)

    return CameraMotion(consensus_prediction).name

def main(args):
    video_folder = args.video_path
    label_file = './labels.txt'
    model_path = './camera_motion_classifier.pth'

    # Create dataset and split into train, eval, and test sets
    dataset = VideoDataset(video_folder, label_file)
    total_size = len(dataset)
    eval_size = test_size = int(0.1 * total_size)
    train_size = total_size - eval_size - test_size

    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CameraMotionClassifier().to(device)

    if args.predict is not None:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            raise FileNotFoundError("No trained model found. Please train the model first.")
        
        predicted_motion = predict_video(model, args.predict, device)
        print(f"Predicted camera motion for video {args.predict}: {predicted_motion}")
        return

    # Load pre-trained model if it exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Resuming training from existing model.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)

    # Evaluate the model on the eval set
    eval_loss, eval_accuracy, eval_predictions, eval_ground_truths = evaluate(model, eval_dataloader, criterion, device)
    print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
    print("Eval Predictions vs Ground Truths:")
    for pred, gt in zip(eval_predictions, eval_ground_truths):
        print(f"Prediction: {CameraMotion(pred).name}, Ground Truth: {CameraMotion(gt).name}")

    # Evaluate the model on the test set
    test_loss, test_accuracy, _, _ = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Motion Classifier')
    parser.add_argument('--video_path', type=str, default='./videos', help='Path to video folder')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--predict', type=str, default=None, help='Path to a video to predict camera motion')
    args = parser.parse_args()

    main(args)