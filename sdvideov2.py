
import os
import argparse
import torch
from typing import List, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import make_image_grid, export_to_video
from dataclasses import dataclass

@dataclass
class DiffusionStep:
    """Store information for each diffusion timestep"""
    timestep: int
    input_latents: torch.Tensor

class LatentTracker:
    """Track and visualize latents during diffusion process"""
    def __init__(self):
        self.steps: List[DiffusionStep] = []
        
    def add_step(self, timestep: int, input_latents: torch.Tensor):
        self.steps.append(
            DiffusionStep(
                timestep=timestep,
                input_latents=input_latents.detach().cpu()
            )
        )
    
    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents for visualization"""
        latents = latents.cpu()
        
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
    
    def plot_latents(self, save_path: Optional[str] = None, num_timesteps: int = 5):
        """Visualize the progression of latents as images"""
        input_latents = torch.cat([step.input_latents for step in self.steps])
        
        # Create visualization grid
        step_size = max(len(self.steps) // num_timesteps, 1)
        selected_indices = list(range(0, len(self.steps), step_size))[:num_timesteps]
        
        selected_latents = [input_latents[idx:idx+1] for idx in selected_indices]
        selected_latents = torch.cat(selected_latents, dim=0)
        
        # Normalize for visualization
        normalized = self._normalize_latents(selected_latents)
        vis_latents = normalized[:, :3].clamp(0, 1)
        
        # Create figure
        plt.figure(figsize=(2*num_timesteps, 4))
        for i in range(num_timesteps):
            plt.subplot(1, num_timesteps, i+1)
            plt.imshow(vis_latents[i].permute(1, 2, 0))
            plt.title(f'Step {selected_indices[i]}')
            plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

class EnhancedVideoPipeline:
    def __init__(self, 
                model_id="stabilityai/stable-diffusion-2-1", 
                controlnet=None, 
                seed=1887,
                fix_seed=False,
                use_cross_frame_attn=False):
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
        self.fix_seed = fix_seed
        self.frame_latents: Dict[int, List[torch.Tensor]] = {}
        self.frame_trackers: Dict[int, LatentTracker] = {}

        self.image_encoder = self.pipeline.vae

        if use_cross_frame_attn:
            from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor, CrossFrameAttnProcessor2_0
            # Apply cross-frame attention processor to UNet and ControlNet
            #cross_frame_processor = CrossFrameAttnProcessor(unet_chunk_size=2)
            #cross_frame_processor = CrossFrameAttnProcessor(batch_size=2)
            cross_frame_processor = CrossFrameAttnProcessor2_0(batch_size=2)
            self.pipeline.unet.set_attn_processor(cross_frame_processor)
            self.pipeline.controlnet.set_attn_processor(cross_frame_processor)

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
        image = (image - 0.5) * 2.0  # Scale to -1 and 1
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32).to(self.device)
        
        # Encode image to latents
        with torch.no_grad():
            image_latents = self.image_encoder.encode(image).latent_dist.sample()
            image_latents = image_latents * 0.18215  # Scale factor from SD
        
        return image_latents
    
    def _create_denoising_callback(self, 
                                   frame_idx: int, 
                                   latent_copy_mode: str = 'disabled', 
                                   reference_latents: Optional[List[torch.Tensor]] = None,
                                   previous_frame_latents: Optional[List[torch.Tensor]] = None,
                                   consistency_strength: float = 0.5,
                                   copy_timestmap: int = 15,
                                   raw_img_latents: Optional[torch.Tensor] = None,
                                   raw_img_latents_blend: float = 0.5) -> callable:
        tracker = LatentTracker()
        self.frame_trackers[frame_idx] = tracker

        def callback(pipe, i: int, t: int, callback_kwargs):
            latents = callback_kwargs['latents']

            if raw_img_latents is not None and 15 <= i <= 15:
                latents = raw_img_latents_blend * raw_img_latents + (1 - raw_img_latents_blend) * latents
                tracker.add_step(i, latents)
                return {'latents': latents}
            
            if reference_latents and latent_copy_mode != 'disabled' and i == copy_timestmap:
                # Choose reference latents based on mode
                if latent_copy_mode == 'first_frame':
                    ref_latents = reference_latents[i]
                elif latent_copy_mode == 'previous_frame':
                    ref_latents = previous_frame_latents[i]
                
                # Blend latents
                blended_latents = (
                    (1 - consistency_strength) * latents +
                    consistency_strength * ref_latents
                )
                tracker.add_step(i, blended_latents)
                return {'latents': blended_latents}
            else:
                tracker.add_step(i, latents)
                return {'latents': latents}
        
        return callback
    
    def generate_frame(self, 
                        prompt: str, 
                        negative_prompts: str,
                        pose_image, 
                        frame_idx: int,
                        raw_image = None,
                       latent_copy_mode: str = 'disabled',
                       reference_latents: Optional[List[torch.Tensor]] = None,
                       previous_frame_latents: Optional[List[torch.Tensor]] = None,
                       consistency_strength: float = 0.5,
                       copy_timestmap: int = 15,
                       num_inference_steps: int = 50,
                       image_conditioning_strength: float = 0.5) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        if raw_image is not None:
            raw_img_latents = self.encode_image_to_latents(raw_image)

        callback = self._create_denoising_callback(
            frame_idx, 
            latent_copy_mode, 
            reference_latents, 
            previous_frame_latents,
            consistency_strength,
            copy_timestmap,
            raw_img_latents=raw_img_latents if raw_image is not None else None,
            raw_img_latents_blend=image_conditioning_strength
        )
        
        if self.fix_seed:
            self.rand_generator.manual_seed(self.seed)
        result = self.pipeline(
            prompt,
            pose_image,
            negative_prompts=negative_prompts,
            generator=self.rand_generator,
            num_inference_steps=num_inference_steps,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=['latents'],
            return_dict=True
        )
        
        # Store frame latents
        #frame_latents = [step.clone().detach() for step in result.repeat_intermediate_latents]
        frame_latents = [step.input_latents.to(self.device) for step in self.frame_trackers[frame_idx].steps]
        self.frame_latents[frame_idx] = frame_latents
        
        return result.images[0], frame_latents
    
    def generate_video(self, prompt: str, negative_prompts: str,
                        pose_sequence: List,
                        raw_img_sequence: Optional[List[Image.Image]] = None,
                       latent_copy_mode: str = 'disabled',
                       consistency_strength: float = 0.5,
                       copy_timestmap: int = 15,
                       num_inference_steps: int = 50,
                       image_conditioning_strength: float = 0.5) -> List:
        # Validate latent copy mode
        if latent_copy_mode not in ['disabled', 'first_frame', 'previous_frame']:
            raise ValueError("latent_copy_mode must be 'disabled', 'first_frame', or 'previous_frame'")
        
        # First frame generation
        first_frame, first_latents = self.generate_frame(
            prompt, negative_prompts,
            pose_sequence[0], 
            frame_idx=0,
            raw_image=raw_img_sequence[0] if raw_img_sequence else None, 
            copy_timestmap=copy_timestmap,
            num_inference_steps=num_inference_steps,
            image_conditioning_strength=image_conditioning_strength
        )
        
        frames = [first_frame]
        
        # Generate subsequent frames
        for idx, pose in enumerate(pose_sequence[1:], 1):
            frame, _ = self.generate_frame(
                prompt, negative_prompts,
                pose, 
                frame_idx=idx,
                raw_image=raw_img_sequence[idx] if raw_img_sequence else None,
                latent_copy_mode=latent_copy_mode,
                reference_latents=first_latents,
                previous_frame_latents=self.frame_latents[idx-1],
                consistency_strength=consistency_strength,
                copy_timestmap=copy_timestmap,
                num_inference_steps=num_inference_steps,
                image_conditioning_strength=image_conditioning_strength,
            )
            frames.append(frame)
        
        return frames


def run_demo(pipe, out_name, seq_len=3, original_images=None, **kwargs):
    added_prompt = (
                    'best quality, extremely detailed, realistic.'
    )
    negative_prompts =  (
                        'longbody, low res, bad anatomy, bad hands, missing fingers, bad figure, '
                        'bad proportions, bad shape, bad silhouette, bad form, bad structure, bad model, '
                        'impossible pose, unnatural pose, awkward pose, bad pose, bad posture, '
                        'extra golf clubs,'
                          'twisted body, broken body, bad pose, bad posture, missing limbs, bad anatomy, '
                            'bad quality, blurry, out of focus, low quality, '
                            'bad composition, bad framing, bad angle, '
    )
    prompt = [
        (
            'A professional male golfer in a red polo shirt and white trousers, holding a golf club in his hands. '
            'The golfer is in the process of a swing seriously. The club is in motion, and its direction is aligned with the arm.'
            'Front-facing view,'
        ) 
        ]
    prompt = [p+added_prompt for p in prompt]
    pose_images = []
    src_dir = "openpose/output_frames"
    inputs = os.listdir(src_dir)
    inputs.sort()
    for file in inputs[:seq_len]:
        control_image = Image.open(src_dir+"/"+file).convert("RGB")
        control_image = control_image.resize((512, 512))  # Resize for compatibility\n",
        pose_images.append(control_image)

    frames = pipe.generate_video(prompt, 
                                negative_prompts, 
                                pose_images, 
                                raw_img_sequence=original_images, 
                                **kwargs)
    pose_images += frames
        
    img = make_image_grid(pose_images, rows=2, cols=len(pose_images)//2)
    export_to_video(frames, out_name+'.mp4', fps=10)
    img.save(out_name+'.png')


def latent_fuse_demo(args):
    controlnet = ControlNetModel.from_pretrained(
        "/home/ubuntu/diffusion/new_controlnet-pose-sd21-diffusers",
        torch_dtype=torch.float32,
        device='cuda',
        local_files_only=True)

    pipe = EnhancedVideoPipeline(model_id="stabilityai/stable-diffusion-2-1", 
                                    controlnet=controlnet,
                                    seed=args.seed,
                                    fix_seed=args.fix_seed)

    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    original_images = None
    if args.original_images_dir:
        original_images_paths = sorted(os.listdir(args.original_images_dir))[:args.num_frames]
        original_images = [
            Image.open(os.path.join(args.original_images_dir, img)).convert("RGB")
            for img in original_images_paths
        ]
    
    run_demo(pipe, 
            os.path.join(save_dir, 
                f'{args.img_prefix}_seed{args.seed}_mode{args.latent_copy_mode}_strength{args.consistency_strength}_copy{args.copy_timestamp}_imgCondStrength{args.image_conditioning_strength}'),
            seq_len=args.num_frames,
            original_images=original_images,
            consistency_strength=args.consistency_strength,
            copy_timestmap=args.copy_timestamp,
            latent_copy_mode=args.latent_copy_mode,
            image_conditioning_strength=args.image_conditioning_strength,
            )

    
    # Debug
    #pipe.visualize_latents(save_dir=save_dir, num_timesteps=50)
    #for i, tracker in pipe.frame_trackers.items():
    #    tracker.save_video(os.path.join(save_dir, f"latent_evolution_frame_{i}.mp4"), fps=10)

def argparser():
    parser = argparse.ArgumentParser(description="Generate video from text prompt")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1", help="Model ID for diffusion model")
    parser.add_argument("--controlnet_path", type=str, default="/home/ubuntu/diffusion/new_controlnet-pose-sd21-diffusers", help="Path to controlnet model")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for generation")
    parser.add_argument("--fix_seed", action="store_true", help="Fix random seed for generation")

    # general parameters
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for each frame")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")

    parser.add_argument("--img_prefix", type=str, default="", help="Prefix for output images")
    parser.add_argument("--output_dir", type=str, default="./latent_visualizations/1207", help="Output directory for generated video")

    # control net parameters
    parser.add_argument("--latent_copy_mode", type=str, 
                        default="previous_frame",
                        choices=["disabled", "first_frame", "previous_frame"],
                        help="Mode for copying latents: 'disabled', 'first_frame', or 'previous_frame'")
    parser.add_argument("--copy_timestamp", type=int, default=8, help="Timestamp to copy latents from reference frame")
    parser.add_argument("--consistency_strength", type=float, default=1.0, help="Consistency strength for latent blending")

    # New arguments for image-based conditioning
    parser.add_argument("--original_images_dir", type=str, default=None, 
                        help="Directory containing original images for latent conditioning")
    parser.add_argument("--image_conditioning_strength", type=float, default=1.0, 
                        help="Strength of image conditioning in latent space")

    # 
    parser.add_argument("--use_cross_frame_attn", action="store_true", help="Use cross-frame attention in UNet and ControlNet")


    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    latent_fuse_demo(args)