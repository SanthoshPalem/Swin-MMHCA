import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.nn.functional as F

from src.models.swin_mmhca import SwinMMHCA
from src.data.dataloader import MultiModalSuperResDataset
from src.models.options import get_args
from src.models.edsr_nav import EDSR_Nav

def train(args):
    start_time = time.time()

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.n_inputs == 1:
        modalities = ['T2']
    else:
        modalities = ['T1', 'T2', 'PD']

    train_dataset = MultiModalSuperResDataset(
        dataset_root=args.dataset_root,
        modalities=modalities,
        scale_factor=args.scale_factor,
        transform=ToTensor(),
        train=True
    )
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = SwinMMHCA(
        n_inputs=args.n_inputs,
        scale=args.scale_factor
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (lr_images, hr_image) in enumerate(dataloader):
            if args.n_inputs == 1:
                lr_images = lr_images[0].unsqueeze(0)
            
            lr_images = [img.to(device) for img in lr_images]
            hr_image = hr_image.to(device)

            optimizer.zero_grad()
            outputs = model(lr_images if args.n_inputs > 1 else lr_images[0])
            loss = criterion(outputs, hr_image)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / args.log_interval:.4f}')
                running_loss = 0.0
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, 'swin_mmhca.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time for {args.epochs} epochs: {elapsed_time:.2f} seconds")

def evaluate(args):
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.model_type == 'EDSR_Nav':
        modalities = ['T2', 'PD']
        n_inputs = 2
    else:
        modalities = ['T1', 'T2', 'PD']
        n_inputs = 3 if args.n_inputs > 1 else 1
        
    val_dataset = MultiModalSuperResDataset(
        dataset_root=args.dataset_root,
        modalities=modalities,
        scale_factor=args.scale_factor,
        transform=ToTensor(),
        train=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model_type == 'SwinMMHCA':
        model = SwinMMHCA(
            n_inputs=n_inputs,
            scale=args.scale_factor
        ).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    elif args.model_type == 'EDSR_Nav':
        edsr_args = get_args()
        edsr_args.n_resblocks = 16
        edsr_args.n_feats = 64
        edsr_args.scale = [args.scale_factor]
        edsr_args.n_colors = 1
        edsr_args.res_scale = 0.1
        edsr_args.shift_mean = False
        edsr_args.use_nav = True
        edsr_args.use_mhca_2 = True
        edsr_args.use_mhca_3 = False
        edsr_args.ratio = 0.5
        edsr_args.use_attention_resblock = True
        
        model = EDSR_Nav(edsr_args).to(device)
        model.load_state_dict(torch.load(args.edsr_checkpoint_path, map_location=device))
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.eval()

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    with torch.no_grad():
        for lr_images, hr_image in val_dataloader:
            if n_inputs == 1:
                lr_images = lr_images[0].unsqueeze(0)
            
            lr_images = [img.to(device) for img in lr_images]
            hr_image = hr_image.to(device)

            outputs = model(lr_images if n_inputs > 1 else lr_images[0])

            total_psnr += psnr(outputs, hr_image)
            total_ssim += ssim(outputs, hr_image)
            total_lpips += lpips_fn(outputs * 2 - 1, hr_image * 2 - 1).mean()

    avg_psnr = total_psnr / len(val_dataloader)
    avg_ssim = total_ssim / len(val_dataloader)
    avg_lpips = total_lpips / len(val_dataloader)

    print(f'Results for {args.model_type}:')
    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average LPIPS: {avg_lpips:.4f}')
    
def visualize(args):
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    swin_model = SwinMMHCA(
        n_inputs=3,
        scale=args.scale_factor
    ).to(device)
    swin_model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    swin_model.eval()

    edsr_args = get_args()
    edsr_args.n_resblocks = 16
    edsr_args.n_feats = 64
    edsr_args.scale = [args.scale_factor]
    edsr_args.n_colors = 1
    edsr_args.res_scale = 0.1
    edsr_args.shift_mean = False
    edsr_args.use_nav = True
    edsr_args.use_mhca_2 = True
    edsr_args.use_mhca_3 = False
    edsr_args.ratio = 0.5
    edsr_args.use_attention_resblock = True
    
    edsr_model = EDSR_Nav(edsr_args).to(device)
    edsr_model.load_state_dict(torch.load(args.edsr_checkpoint_path, map_location=device))
    edsr_model.eval()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with torch.no_grad():
        if args.use_test_samples:
            with open(os.path.join(args.test_sample_path, f'HR_T2_500_0_x{args.scale_factor}.pt'), 'rb') as _f:
                t2w_lr = np.expand_dims(pickle.load(_f), axis=(0, 1))
            with open(os.path.join(args.test_sample_path, f'HR_PD_500_0_x{args.scale_factor}.pt'), 'rb') as _f:
                pd_lr = np.expand_dims(pickle.load(_f),  axis=(0, 1))

            dummy_t1_lr = np.zeros_like(t2w_lr)

            lr_images_swin = [
                torch.from_numpy(dummy_t1_lr).float(),
                torch.from_numpy(t2w_lr).float(),
                torch.from_numpy(pd_lr).float()
            ]
            
            lr_images_swin = [F.interpolate(img, size=(64, 64), mode='bicubic', align_corners=False).to(device) for img in lr_images_swin]

            lr_images_edsr = [
                F.interpolate(torch.from_numpy(t2w_lr).float(), size=(64, 64), mode='bicubic', align_corners=False).to(device),
                F.interpolate(torch.from_numpy(pd_lr).float(), size=(64, 64), mode='bicubic', align_corners=False).to(device)
            ]
            
            swin_output = swin_model(lr_images_swin)
            edsr_output = edsr_model(lr_images_edsr)

            lr_display = F.interpolate(torch.from_numpy(t2w_lr).float(), size=(64, 64), mode='bicubic', align_corners=False).squeeze().numpy()
            swin_display = swin_output.squeeze().cpu().numpy()
            edsr_display = edsr_output.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Visual Comparison on Test Sample', fontsize=16)
            
            axes[0].imshow(lr_display, cmap='gray')
            axes[0].set_title('Low-Resolution Input (T2)')
            axes[0].axis('off')

            axes[1].imshow(edsr_display, cmap='gray')
            axes[1].set_title('EDSR_Nav Output')
            axes[1].axis('off')

            axes[2].imshow(swin_display, cmap='gray')
            axes[2].set_title('SwinMMHCA Output')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(args.save_dir, 'test_sample_comparison.png')
            plt.savefig(save_path)
            plt.close(fig)
            
            print(f"Saved comparison for test sample to {save_path}")

        else:
            val_dataset = MultiModalSuperResDataset(
                dataset_root=args.dataset_root,
                modalities=['T1', 'T2', 'PD'],
                scale_factor=args.scale_factor,
                transform=ToTensor(),
                train=False,
                shuffle=False
            )
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            for i, (lr_images, hr_image) in enumerate(val_dataloader):
                if i >= args.num_samples:
                    break
                    
                lr_images_swin = [img.to(device) for img in lr_images]
                lr_images_edsr = [lr_images[1].to(device), lr_images[2].to(device)]
                hr_image = hr_image.to(device)

                swin_output = swin_model(lr_images_swin)
                edsr_output = edsr_model(lr_images_edsr)
                
                lr_display = lr_images[1].squeeze().cpu().numpy()
                hr_display = hr_image.squeeze().cpu().numpy()
                swin_display = swin_output.squeeze().cpu().numpy()
                edsr_display = edsr_output.squeeze().cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                fig.suptitle(f'Visual Comparison for Sample {i+1}', fontsize=16)
                
                axes[0].imshow(lr_display, cmap='gray')
                axes[0].set_title('Low-Resolution Input')
                axes[0].axis('off')

                axes[1].imshow(edsr_display, cmap='gray')
                axes[1].set_title('EDSR_Nav Output')
                axes[1].axis('off')

                axes[2].imshow(swin_display, cmap='gray')
                axes[2].set_title('SwinMMHCA Output')
                axes[2].axis('off')

                axes[3].imshow(hr_display, cmap='gray')
                axes[3].set_title('High-Resolution Ground Truth')
                axes[3].axis('off')

                plt.tight_layout()
                save_path = os.path.join(args.save_dir, f'comparison_sample_{i+1}.png')
                plt.savefig(save_path)
                plt.close(fig)
                
                print(f"Saved comparison for sample {i+1} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='SwinMMHCA for Super-Resolution')

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'visualize'], help='Operating mode')

    # Hardware specifications
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')

    # Data specifications
    parser.add_argument('--dataset_root', type=str, default='datasets', help='root directory of the dataset')
    parser.add_argument('--scale_factor', type=int, default=4, help='super-resolution scale factor')
    parser.add_argument('--n_inputs', type=int, default=3, help='number of input modalities (1 for single-input, >1 for multi-input)')

    # Model specifications
    parser.add_argument('--model_type', type=str, default='SwinMMHCA', choices=['SwinMMHCA', 'EDSR_Nav'], help='type of model to use')
    parser.add_argument('--checkpoint_path', type=str, default='pretrained_models/swin_mmhca.pth', help='path to the model checkpoint')

    # Training specifications
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--log_interval', type=int, default=10, help='interval for printing training loss')
    parser.add_argument('--save_dir', type=str, default='results', help='directory to save the trained model')

    # Evaluation specifications
    parser.add_argument('--edsr_checkpoint_path', type=str, default='../MHCA-main/edsr/pretrained_models/model_multi_input_IXI_x4.pt', help='Path to the EDSR_Nav model checkpoint')

    # Visualization specifications
    parser.add_argument('--num_samples', type=int, default=3, help='number of samples to visualize')
    parser.add_argument('--use_test_samples', action='store_true', help='Use the test samples from MHCA-main')
    parser.add_argument('--test_sample_path', type=str, default='../MHCA-main/edsr/test_samples', help='Path to the test samples directory')


    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'visualize':
        visualize(args)

if __name__ == '__main__':
    main()