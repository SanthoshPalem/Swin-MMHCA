import os
import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
from PIL import Image

class MultiModalSuperResDataset(data.Dataset):
    def __init__(self, dataset_root, modalities=['T1', 'T2', 'PD'], scale_factor=4, transform=None, split='train', shuffle=True):
        super(MultiModalSuperResDataset, self).__init__()
        
        self.dataset_root = dataset_root
        self.modalities = modalities
        self.scale_factor = scale_factor
        self.transform = transform
        
        all_samples = self._scan_dataset()
        if shuffle:
            np.random.shuffle(all_samples)
            
        # Per the paper: 500 train, 6 validation, 70 test
        train_end = 500
        val_end = 500 + 6
        
        if split == 'train':
            self.samples = all_samples[:train_end]
        elif split == 'validation':
            self.samples = all_samples[train_end:val_end]
        elif split == 'test':
            self.samples = all_samples[val_end:val_end + 70]
        else:
            raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'validation', 'test'.")

    def _scan_dataset(self):
        samples = []
        modality_roots = {mod: os.path.join(self.dataset_root, f'IXI-{mod}') for mod in self.modalities}
        
        primary_modality = self.modalities[0]
        primary_modality_root = modality_roots[primary_modality]
        if not os.path.isdir(primary_modality_root):
            raise FileNotFoundError(f"Directory not found: {primary_modality_root}")

        for filename in os.listdir(primary_modality_root):
            if filename.endswith(f'-{primary_modality}.nii.gz'):
                base_name = filename.replace(f'-{primary_modality}.nii.gz', '')
                
                sample_files = {}
                all_files_exist = True
                for mod in self.modalities:
                    expected_filename = f"{base_name}-{mod}.nii.gz"
                    file_path = os.path.join(modality_roots[mod], expected_filename)
                    if os.path.exists(file_path):
                        sample_files[mod] = file_path
                    else:
                        all_files_exist = False
                        break
                
                if all_files_exist:
                    samples.append(sample_files)
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_files = self.samples[index]
        
        lr_images = []
        hr_image = None
        
        # The target modality for super-resolution is T2
        target_modality = 'T2'
        
        for mod in self.modalities:
            file_path = sample_files[mod]
            
            # Load the NIfTI image
            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata()
            
            # Take the central slice of the 3D volume
            central_slice_idx = img_data.shape[2] // 2
            slice_data = img_data[:, :, central_slice_idx]
            
            # Normalize and convert to PIL Image
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            slice_data = (slice_data * 255).astype(np.uint8)
            img_pil = Image.fromarray(slice_data).convert('L') # Grayscale
            
            # Generate LR image & HR image and resize
            width, height = img_pil.size
            lr_img_pil = img_pil.resize((width // self.scale_factor, height // self.scale_factor), Image.BICUBIC)
            
            # Resize for the model
            lr_img_pil = lr_img_pil.resize((64, 64), Image.BICUBIC)
            
            if self.transform:
                lr_img = self.transform(lr_img_pil)

            lr_images.append(lr_img)

            if mod == target_modality:
                # Resize HR image for the model output
                hr_img_pil = img_pil.resize((256, 256), Image.BICUBIC)
                if self.transform:
                    hr_image = self.transform(hr_img_pil)
                else:
                    hr_image = hr_img_pil
        
        return lr_images, hr_image

if __name__ == '__main__':
    from torchvision.transforms import ToTensor
    import os
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Construct the path to the dataset
    dataset_root = os.path.join(project_root, 'datasets')
    
    # Example of how to use the dataset
    train_dataset = MultiModalSuperResDataset(
        dataset_root=dataset_root,
        modalities=['T1', 'T2', 'PD'],
        scale_factor=4,
        transform=ToTensor(),
        train=True
    )
    
    val_dataset = MultiModalSuperResDataset(
        dataset_root=dataset_root,
        modalities=['T1', 'T2', 'PD'],
        scale_factor=4,
        transform=ToTensor(),
        train=False
    )
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    if len(val_dataset) > 0:
        lr_imgs, hr_img = val_dataset[0]
        
        print(f"Number of LR images: {len(lr_imgs)}")
        print(f"LR image shape: {lr_imgs[0].shape}")
        print(f"HR image shape: {hr_img.shape}")
    else:
        print("No samples found in the validation dataset.")