import torch
from torch import nn
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from PIL import Image

"""
This code is for loading an artificial 
intelligence model that enhances super resolution, 
and it is used to upgrade a low-quality 
720p image to 1080p and improve its quality.
"""


# Define the Enhanced Deep Super-Resolution (EDSR) model
class EDSR(nn.Module):
    def __init__(self, scale_factor):
        super(EDSR, self).__init__()
        # Use ReLU for the activation function
        self.relu = nn.ReLU(inplace=True)

        # Define the layers of the model
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        # Upsampling layer with bicubic interpolation
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True)

    def forward(self, x):
        # Define the forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)
        return x


# Function to load a trained model from a file
def load_model(model_path):
    model = EDSR(scale_factor=1.5)  # Example scale factor
    # Load the model's weights from the specified file
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # Set the model to evaluation mode
    model.eval()
    return model


# Function to process an image using the model
def process_image(model, image_path, output_filename):
    # Load the image from the given path
    input_image = Image.open(image_path).convert('RGB')

    # Resize the image to 1280x720 pixels
    transform = Compose([Resize((720, 1280)), ToTensor()])
    input_image = transform(input_image).unsqueeze(0)

    # Process the image with the model
    with torch.no_grad():
        output_image = model(input_image)

    # Save the processed image to the specified filename
    save_image(output_image, output_filename)


# Function to save a tensor as an image file
def save_image(tensor, filename):
    image = tensor.cpu().clone()  # Copy the tensor to CPU
    image = image.squeeze(0)  # Remove batch dimension
    image = ToPILImage()(image)  # Convert the tensor to a PIL image
    image.convert("RGB").save(filename)  # Save the image


# Load your model
model = load_model('Example_EDSR_Model.pth')  # Load the trained EDSR model

# Specify the path of the photo to be processed and the name of the output file
image_path = 'test_image.jpg'  # Path of the photo
output_filename = 'test_image_output.jpg'  # Name of the file to be saved

# Process the photo
process_image(model, image_path, output_filename)  # Apply the model to the photo
