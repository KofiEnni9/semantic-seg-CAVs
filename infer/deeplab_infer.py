import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from network.modeling import deeplabv3plus_resnet50
from network._deeplab import convert_to_separable_conv

class SegmentationInference:
    def __init__(self, model_path, num_classes=7, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Initialize model
        self.model = deeplabv3plus_resnet50(num_classes=num_classes)
        convert_to_separable_conv(self.model.classifier)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image_path):
        """Load and preprocess an image for inference"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        # Store original size for later
        original_size = image.size
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device), original_size
    
    def predict(self, image_tensor):
        """Run inference on the preprocessed image"""
        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, tuple):
                output = output[1]  # Get main output if model returns auxiliary outputs
            pred = torch.argmax(output, dim=1)
            return pred
    
    def create_colored_mask(self, mask, original_size):
        """Convert prediction mask to colored image"""
        class_colors = {
            0: (0, 0, 0),       # black
            1: (208, 254, 157), # Light green
            2: (59, 93, 4),     # Dark green
            3: (155, 155, 154), # Gray
            4: (138, 87, 42),   # Brown
            5: (183, 21, 123),  # Pink
            6: (73, 143, 225)   # Blue
        }
        
        # Convert to numpy and resize to original image size
        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_np = cv2.resize(mask_np, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask
        colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        for class_idx, color in class_colors.items():
            colored_mask[mask_np == class_idx] = color
            
        return colored_mask

def process_folder(input_folder, output_folder, model_path, num_classes=7):
    """Process all images in a folder and save segmentation results"""
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize inference class
    inferencer = SegmentationInference(model_path, num_classes)
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if os.path.splitext(filename)[1].lower() in valid_extensions:
            print(f"Processing {filename}...")
            
            # Full paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"segmented_{filename}")
            
            try:
                # Preprocess image
                image_tensor, original_size = inferencer.preprocess_image(input_path)
                
                # Run inference
                prediction = inferencer.predict(image_tensor)
                
                # Create colored segmentation mask
                colored_mask = inferencer.create_colored_mask(prediction[0], original_size)
                
                # Save result
                cv2.imwrite(output_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                
                print(f"Saved segmentation result to {output_path}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "data/infer/input"  # Folder containing images to process
    OUTPUT_FOLDER = "data/infer/output"  # Folder to save segmentation results
    MODEL_PATH = "best_model_yet.pth"    # Path to your trained model weights
    NUM_CLASSES = 7                      # Number of segmentation classes
    
    # Process all images in the folder
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH, NUM_CLASSES)