import os
from datetime import datetime
import torch
from torch import nn
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from PIL import Image

"""
This code trains an artificial intelligence model 
for super-resolution using high-quality and low-quality 
photos from the folders '720p_frames' and '1080p_frames', 
utilizing the torch library. After training, 
it saves the model and attempts to upscale a 720p photo to 1080p.
"""

# Configure the device (use GPU if available for faster training and inference)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class EDSR(nn.Module):
    def __init__(self, scale_factor):
        super(EDSR, self).__init__()
        # Building the EDSR model layers.
        # First and second convolutions increase feature extraction capability.
        # Upsample layer enlarges the image, and the final convolution produces the output.
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True)

    def forward(self, x):
        # Defining the forward pass through the network.
        # The network uses ReLU activations and a bicubic upsampling method.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)
        return x


def load_model(path='model.pth'):
    # Function to load the EDSR model. If the model file is not found, it starts with a new model.
    model = EDSR(scale_factor).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model loaded from: " + path)
    except Exception as e:
        print("Failed to load model. Starting with a new model. Error: ", e)
    return model


def load_low_res_image(image_path):
    # Load and convert a low-resolution image to a PyTorch tensor.
    image = Image.open(image_path).convert('RGB')
    return ToTensor()(image)


def load_high_res_image(image_path, transform):
    # Load a high-resolution image, apply transformations, and convert to a PyTorch tensor.
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def save_model(model, path='model.pth'):
    # Save the state of the model to the specified file path.
    torch.save(model.state_dict(), path)


def save_image(tensor, filename):
    # Convert a PyTorch tensor to an image and save it to the specified file path.
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = ToPILImage()(image)
    image.save(filename)


def process_image(model, image_path, output_filename):
    # Process an image file using the trained model and save the output.
    # This includes loading the image, applying the model, and saving the output.
    input_image = load_low_res_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output_image = model(input_image)
    save_image(output_image, output_filename)


def train_model(model, criterion, optimizer, train_loader, num_epochs=25):
    # Main function for training the model.
    # It iterates over the training dataset and optimizes the model's weights.
    for epoch in range(num_epochs):
        current_time = datetime.now().strftime("%H:%M")
        print(f"Epoch {epoch} starting, Time: {current_time}")
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, low_res_folder, high_res_folder, transform=None):
        # Custom dataset class to handle pairs of low and high-resolution images.
        # This is useful for loading and transforming data for the EDSR model.
        self.low_res_folder = low_res_folder
        self.high_res_folder = high_res_folder
        self.transform = transform
        self.image_filenames = [x for x in os.listdir(low_res_folder) if is_image_file(x)]

    def __getitem__(self, index):
        # Retrieve a pair of low and high-resolution images by index.
        img_name = self.image_filenames[index]
        low_res_path = os.path.join(self.low_res_folder, img_name)
        high_res_path = os.path.join(self.high_res_folder, img_name)
        low_res = load_low_res_image(low_res_path)
        high_res = load_high_res_image(high_res_path, self.transform)
        return low_res, high_res

    def __len__(self):
        # Return the total number of image pairs in the dataset.
        return len(self.image_filenames)


# Initialize and prepare for model training

# Set the scaling factor for image upscaling.
scale_factor = 1.5

# Load the EDSR model. If 'edsr_model.pth' exists, it loads this pre-trained model; otherwise, it initializes a new
# model.
model = load_model("Example_EDSR_Model1.pth")

# Define the loss function as Mean Squared Error Loss, suitable for regression tasks like image super-resolution.
criterion = nn.MSELoss().to(device)

# Define the optimizer as Adam, which is an effective optimization algorithm for training deep learning models.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Prepare the image data loaders. The dataset consists of pairs of low and high-resolution images. The low-resolution
# images are located in '720p_frames/', and their corresponding high-resolution versions are in '1080p_frames/'. The
# images are resized to 1080x1920 pixels with antialiasing to maintain image quality, and then converted to PyTorch
# tensors.
transform = Compose([Resize((1080, 1920), antialias=True), ToTensor()])
train_dataset = ImageDataset(low_res_folder='720p_frames/', high_res_folder='1080p_frames/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print("Data loaded, training started")

# Start the training process. The model is trained for 5 epochs.
# An epoch is a complete pass through the entire training dataset.
train_model(model, criterion, optimizer, train_loader, num_epochs=5)

try:
    # Test the trained model by enhancing a low-resolution sample image. The image '720p/vid1_frame_667.jpg' is
    # processed, and the resulting high-resolution image is saved as 'output.png'.
    process_image(model, 'test_image.jpg', 'test_image_output.png')
except Exception as e:
    # In case of an error (e.g., file not found, processing error), it is printed to the console.
    print("Error in saving image: " + str(e))

# Finally, save the updated state of the model to a new file 'edsr_model1.pth'.
# This allows the model's learned weights to be reused without needing to retrain from scratch.
save_model(model, 'EDSR_Model1.pth')
