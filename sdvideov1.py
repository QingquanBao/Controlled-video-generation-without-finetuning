import os
from PIL import Image
import torch
import einops
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math
import numpy as np

import torch.nn.functional as F
from diffusers.utils import BaseOutput
from diffusers.utils import make_image_grid
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor, CrossFrameAttnProcessor2_0
import matplotlib.pyplot as plt

@dataclass
class DiffusionStep:
    """Store information for each diffusion timestep"""
    timestep: int
    input_latents: torch.Tensor
    

class LatentTracker:
    """Track and visualize latents during diffusion process"""
    def __init__(self):
        self.steps: List[DiffusionStep] = []
        
    def add_step(self, timestep: int, 
                    input_latents: torch.Tensor, 
                    ):
        self.steps.append(
            DiffusionStep(
                timestep=timestep,
                input_latents=input_latents.detach().cpu(),
            )
        )
    
    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents for visualization"""
        # Ensure we're working with CPU tensors
        latents = latents.cpu()
        
        # Normalize each channel separately
        B, C, H, W = latents.shape
        normalized = torch.zeros_like(latents)
        for c in range(C):
            channel = latents[:, c:c+1]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                normalized[:, c:c+1] = (channel - min_val) / (max_val - min_val)
            else:
                normalized[:, c:c+1] = channel - min_val
        
        return normalized
    
    def _create_latent_grid(self, latents: torch.Tensor, num_timesteps: int) -> torch.Tensor:
        """Create a grid of latent visualizations"""
        # Select evenly spaced timesteps
        step_size = max(len(self.steps) // num_timesteps, 1)
        selected_indices = list(range(0, len(self.steps), step_size))[:num_timesteps]
        
        # Create grid
        selected_latents = []
        for idx in selected_indices:
            selected_latents.append(latents[idx:idx+1])
        
        selected_latents = torch.cat(selected_latents, dim=0)
        
        # Normalize for visualization
        normalized = self._normalize_latents(selected_latents)
        
        # Rearrange channels to create RGB visualization
        # Take first 3 channels and ensure proper range
        vis_latents = normalized[:, :3].clamp(0, 1)
        
        return vis_latents, selected_indices
    
    def plot_latents(self, save_path: Optional[str] = None, num_timesteps: int = 5):
        """Visualize the progression of latents as images"""
        # Stack all latents
        input_latents = torch.cat([step.input_latents for step in self.steps])
        
        # Create visualization grids
        input_grid, input_indices = self._create_latent_grid(input_latents, num_timesteps)
        
        # Create figure
        fig, axes = plt.subplots(2, num_timesteps, figsize=(2*num_timesteps, 4))
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        
        # Plot input latents
        for i in range(num_timesteps):
            axes[0, i].imshow(input_grid[i].permute(1, 2, 0))
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Step {input_indices[i]}')
        axes[0, 0].set_ylabel('Input\nLatents', rotation=0, labelpad=40)
        
        # Add overall title
        plt.suptitle(f"Latent Space Progression {save_path.split('.')[0]}", y=1.05)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def save_video(self, save_path: str, fps: int = 5):
        """Save the latent progression as a video"""
        try:
            import cv2
        except ImportError:
            print("opencv-python is required for video saving. Please install it with pip install opencv-python")
            return
            
        # Stack all latents
        input_latents = torch.cat([step.input_latents for step in self.steps])
        
        # Normalize latents
        input_vis = self._normalize_latents(input_latents)[:, :3]
        
        # Convert to uint8
        input_vis = (input_vis.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        
        # Create video writer
        H, W = input_vis.shape[1:3]
        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (W, H)  # Width doubled to show input and denoised side by side
        )
        
        # Write frames
        for input_frame in input_vis:
            # Convert from RGB to BGR
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
            writer.write(input_frame)

        writer.release()

# Old one but with some new features
class EnhancedVideoPipeline:
    """Enhanced video pipeline with latent visualization capabilities"""
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", controlnet=None, seed=1887):
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            scheduler=self.scheduler,
            torch_dtype=torch.float32
        ).to('cuda')
        self.device = 'cuda'
        self.rand_generator = torch.Generator(device='cuda')
        self.seed = seed

        # Store latent trackers for different frames
        self.frame_trackers: Dict[int, LatentTracker] = {} 
        self.image_encoder = self.pipeline.vae

    def encode_image_to_latents(self, image: Image.Image) -> torch.Tensor:
        """
        Encode an input image to its latent representation.
        
        Args:
            image (Image.Image): Input image to encode
        
        Returns:
            torch.Tensor: Latent representation of the image
        """
        # Preprocess the image
        image = image.convert("RGB")
        image = image.resize((512, 512))  # Resize to match SD input
        
        # Convert to tensor and normalize
        image = np.array(image) / 255.0
        image = (image - 0.5) * 2.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32).to(self.device)
        
        # Encode image to latents
        with torch.no_grad():
            image_latents = self.image_encoder.encode(image).latent_dist.sample()
            image_latents = image_latents * 0.18215  # Scale factor from SD
        
        return image_latents
        
    def _create_denoising_callback(self, 
                                 frame_idx: int, 
                                 use_copying: bool = False, 
                                 use_diff: bool = False,
                                 copy_ts: int = 15,
                                 reference_latents: Optional[List[torch.Tensor]] = None,
                                 consistency_strength: float = 0.5,
                                 raw_img_latents: Optional[torch.Tensor] = None,
                                 mask : Optional[torch.Tensor] = None):
        """Create callback function for tracking latents during denoising"""
        tracker = LatentTracker()
        self.frame_trackers[frame_idx] = tracker
        
        #def callback(i: int, t: int, latents: torch.Tensor):
        def callback(pipe, i: int, t: int, callback_kwargs):
            latents = callback_kwargs['latents']
            
            if use_diff:
                raw_latent_diff = [raw_img_latents[idx] - raw_img_latents[0] for idx in range(1,len(raw_img_latents))]

            if raw_img_latents is not None and mask is not None:
                if use_copying and reference_latents is not None and not use_diff and i <= copy_ts:
                    # Blend with reference latents if copying is enabled
                
                    mask_ = torch.nn.functional.interpolate(mask[frame_idx].unsqueeze(0).unsqueeze(0), size=latents.size()[2:])

                    blended_latents = (
                        (1 - consistency_strength) * latents +
                        consistency_strength * 
                        (reference_latents[i] * (1-mask_) + 
                        mask_ * raw_img_latents[frame_idx])
                    )
                    tracker.add_step(i, latents) #, denoised)
                    #latents = blended_latents
                    return {'latents': blended_latents}
                
                elif reference_latents is None and i <= copy_ts:
                    mask_ = torch.nn.functional.interpolate(mask[frame_idx].unsqueeze(0).unsqueeze(0), size=latents.size()[2:])
                    blended_latents = (
                        (1 - consistency_strength) * latents +
                        consistency_strength * mask_ * raw_img_latents[frame_idx] + 
                        consistency_strength * (1 - mask_) * latents
                    )
                    # Just track the latents for vanilla diffusion
                    tracker.add_step(i, latents) #, denoised)
                    return {'latents': blended_latents}
                else:
                    # Just track the latents for vanilla diffusion
                    tracker.add_step(i, latents) #, denoised)
                    return {'latents': latents}

            if use_diff and reference_latents is not None and i == copy_ts:

                blended_latents = (
                    (1 - consistency_strength) * latents +
                    consistency_strength * (reference_latents[i] + raw_latent_diff[frame_idx-1])
                )
                tracker.add_step(i, latents) #, denoised)
                #latents = blended_latents
                return {'latents': blended_latents}
            elif use_copying and reference_latents is not None and not use_diff and i == copy_ts:
                # Blend with reference latents if copying is enabled
                blended_latents = (
                    (1 - consistency_strength) * latents +
                    consistency_strength * reference_latents[i]
                )
                tracker.add_step(i, latents) #, denoised)
                #latents = blended_latents
                return {'latents': blended_latents}
            else:
                # Just track the latents for vanilla diffusion
                tracker.add_step(i, latents) #, denoised)
                return {'latents': latents}
                
        return callback
        
    def generate_frame(self, prompt: str, pose_image, frame_idx: int,
                      reference_latents: Optional[List[torch.Tensor]] = None,
                      consistency_strength: float = 0.5,
                      num_inference_steps: int = 50,
                      copy_ts: int = 15,
                      use_diff: bool = False,
                      raw_img_latents: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate a single frame with latent tracking"""
        use_copying = reference_latents is not None
        
        callback = self._create_denoising_callback(
            frame_idx, 
            #text_embeddings, control_image,
            use_copying, use_diff, copy_ts,
            reference_latents, consistency_strength,
            raw_img_latents=raw_img_latents,
            mask=mask
        )
        
        self.rand_generator.manual_seed(self.seed)
        result = self.pipeline(
            prompt,
            pose_image,
            generator=self.rand_generator,
            num_inference_steps=num_inference_steps,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=['latents'],
            # callback_on=callback,
            # callback_steps=1,
            return_dict=True
        )
        
        return result.images[0], [step.input_latents.to(self.device) for step in self.frame_trackers[frame_idx].steps]
    
    def generate_video(self, prompt: str, pose_sequence: List,
                      consistency_strength: float = 0.5,
                      num_inference_steps: int = 50,
                      copy_ts: int = 15,
                      use_diff=False,
                      raw_image = None,
                      masks: Optional[List[torch.Tensor]] = None) -> List:
        """Generate video frames with latent visualization"""
        if raw_image is not None:
            raw_img_latents = [self.encode_image_to_latents(raw) for raw in raw_image]

        # copy stuff to cuda memory
        if masks is not None:
            for i in range(len(masks)):
                masks[i] = masks[i].cuda()

        # Generate first frame (vanilla diffusion)
        first_frame, first_latents = self.generate_frame(
            prompt, pose_sequence[0], frame_idx=0,
            consistency_strength=consistency_strength,
            num_inference_steps=num_inference_steps,
            copy_ts=copy_ts,
            use_diff=use_diff,
            raw_img_latents=raw_img_latents,
            mask=masks
        )
        
        frames = [first_frame]
        
        # Generate subsequent frames with latent copying
        for idx, pose in enumerate(pose_sequence[1:], 1):
            frame, _ = self.generate_frame(
                prompt, pose, frame_idx=idx,
                reference_latents=first_latents,
                consistency_strength=consistency_strength,
                num_inference_steps=num_inference_steps,
                copy_ts=copy_ts,
                use_diff=use_diff,
                raw_img_latents=raw_img_latents,
                mask=masks
            )
            frames.append(frame)
        
        return frames
    
    def visualize_latents(self, save_dir: str, num_timesteps: int = 50):
        """Visualize latents for all frames"""
        for frame_idx, tracker in self.frame_trackers.items():
            save_path = f"{save_dir}/frame_{frame_idx}_latents.png"
            tracker.plot_latents(save_path, num_timesteps=num_timesteps)

    
def get_diff(seq_len):
    positive_diff_images = []
    negative_diff_images = []

    # Read images from the folder
    for i in range(min(len(os.listdir("latent_diff"))//2, seq_len-1)):
        positive_diff_image_path = f"latent_diff/positive_diff_{i+1}.npy"
        negative_diff_image_path = f"latent_diff/negative_diff_{i+1}.npy"

        # Check if the files exist
        if os.path.exists(positive_diff_image_path):
            positive_diff_images.append(torch.from_numpy(np.load(positive_diff_image_path).transpose(2,0,1)))
        if os.path.exists(negative_diff_image_path):
            negative_diff_images.append(torch.from_numpy(np.load(negative_diff_image_path).transpose(2,0,1)))

    return positive_diff_images, negative_diff_images

def get_mask(seq_len):
    masks = []

    # Read images from the folder
    for i in range(min(len(os.listdir("masks")), seq_len)):
        mask = f"masks/mask_{i:02}.png"

        # Check if the files exist
        if os.path.exists(mask):
            mask_image = Image.open(mask).convert("L")  # Convert to grayscale
            mask_tensor = torch.from_numpy(np.array(mask_image) / 255.0).float()  # Normalize values to 0 or 1
            masks.append(mask_tensor)
       
    return masks

def run_demo(pipe, out_name, seq_len=3, **kwargs):
    prompt = ['man playing golf with red shirt']
    outputs = []
    src_dir = "openpose/output_frames"
    inputs = os.listdir(src_dir)
    inputs.sort()
    for file in inputs[:seq_len]:
        control_image = Image.open(src_dir+"/"+file).convert("RGB")
        control_image = control_image.resize((512, 512))  # Resize for compatibility\n",
        outputs.append(control_image)
        #out = pipe(prompt, control_image,).images[0]
        #outputs.append(out)

    frames = pipe.generate_video(prompt, outputs, **kwargs)
    outputs += frames
        
    img = make_image_grid(outputs, rows=2, cols=len(outputs)//2)
    img.save(out_name)

def run_demo_with_diff_control(pipe, out_name, seq_len=3, copy_ts=15, **kwargs):
    prompt = ['man playing golf with red shirt']
    outputs = []
    src_dir = "openpose/output_frames"
    inputs = os.listdir(src_dir)
    inputs.sort()
    for idx, file in enumerate(inputs[:seq_len]):
        control_image = Image.open(src_dir+"/"+file).convert("RGB")
        control_image = control_image.resize((512, 512))  # Resize for compatibility\n",
        if idx == 0:
            outputs.append(control_image)
        else:
            black_image = Image.new('RGB', control_image.size, (0, 0, 0))
            outputs.append(black_image)
        #out = pipe(prompt, control_image,).images[0]
        #outputs.append(out)

    original_images = None
    original_images_dir = 'frames'
    if original_images_dir:
        original_images_paths = sorted(os.listdir(original_images_dir))[:seq_len]
        original_images = [
            Image.open(os.path.join(original_images_dir, img)).convert("RGB")
            for img in original_images_paths
        ]

    frames = pipe.generate_video(prompt, outputs, use_diff=True, raw_image=original_images, copy_ts=copy_ts, **kwargs)
    outputs += frames
        
    img = make_image_grid(outputs, rows=2, cols=len(outputs)//2)
    img.save(out_name)

def run_demo_with_masked_latent(pipe, out_name, seq_len=3, copy_ts=15, **kwargs):
    masks = get_mask(seq_len=seq_len)
    original_images = None
    original_images_dir = 'frames'
    if original_images_dir:
        original_images_paths = sorted(os.listdir(original_images_dir))[:seq_len]
        original_images = [
            Image.open(os.path.join(original_images_dir, img)).convert("RGB")
            for img in original_images_paths
        ]
    prompt = ['man playing golf with red shirt']
    outputs = []
    src_dir = "openpose/output_frames"
    inputs = os.listdir(src_dir)
    inputs.sort()
    for file in inputs[:seq_len]:
        control_image = Image.open(src_dir+"/"+file).convert("RGB")
        control_image = control_image.resize((512, 512))  # Resize for compatibility\n",
        outputs.append(control_image)
        #out = pipe(prompt, control_image,).images[0]
        #outputs.append(out)
    frames = pipe.generate_video(prompt, outputs, 
                                 raw_image=original_images,
                                 masks=masks, 
                                 copy_ts=copy_ts,
                                 **kwargs)
    outputs += frames
        
    img = make_image_grid(outputs, rows=2, cols=len(outputs)//2)
    img.save(out_name)


def latent_fuse_demo():
    controlnet = ControlNetModel.from_pretrained(
        "/home/ubuntu/diffusion/new_controlnet-pose-sd21-diffusers",
        torch_dtype=torch.float32,
        device='cuda',
        local_files_only=True)


    pipe = EnhancedVideoPipeline(model_id="stabilityai/stable-diffusion-2-1", 
                                    controlnet=controlnet,)

    # save_dir = "./latent_visualizations/copy_t=15"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # run_demo(pipe, 
    #         os.path.join(save_dir, 'fuse_lat_video_demo_strength-1-fixseed1887.jpg'), 
    #         seq_len=16,
    #         consistency_strength=1.0)
    
    save_dir = "./latent_visualizations/diff_control"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for copy_ts in range(0,50,5):
        run_demo_with_diff_control(pipe, 
                os.path.join(save_dir, f'{copy_ts}.jpg'), 
                seq_len=10,
                consistency_strength=1.0,
                copy_ts=copy_ts)

    # save_dir = "./latent_visualizations/mask_latent"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # for copy_ts in range(5,50,5):
    
    #     run_demo_with_masked_latent(pipe, 
    #             os.path.join(save_dir, f'<{copy_ts}.jpg'), 
    #             seq_len=10,
    #             consistency_strength=1.0,
    #             copy_ts=copy_ts)
    
    # pipe.visualize_latents(save_dir=save_dir, num_timesteps=50)
    # for i, tracker in pipe.frame_trackers.items():
    #    tracker.save_video(os.path.join(save_dir, f"latent_evolution_frame_{i}.mp4"), fps=10)


if __name__ == '__main__':
    latent_fuse_demo()

