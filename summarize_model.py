import torch
from torchsummary import summary
from src.models.swin_mmhca import SwinMMHCA

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Since we are just summarizing, we can use a smaller number of inputs
    model = SwinMMHCA(n_inputs=3).to(device)
    
    print("SwinMMHCA Model Summary:")
    print(model)
    
    print("\n\nSwinMMHCA (single-input) Summary:")
    model_single = SwinMMHCA(n_inputs=1).to(device)
    summary(model_single, (1, 64, 64))

if __name__ == '__main__':
    main()