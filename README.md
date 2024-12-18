## Controlled video generation without finetuning

![demo/demo1.png](demo/demo1.png)
![demo/demo2.png](demo/demo2.jpg)

1. Look at https://huggingface.co/thibaud/controlnet-sd21. This is the controlnet + sd 2.1 checkpoint
2. Convert the `.ckpt` into diffuser directory.
```
python  convert_original_controlnet_to_diffusers.py --checkpoint_path control_v11p_sd21_openpose/cont
rol_v11p_sd21_openpose.ckpt  --original_config_file ./cldm_v21.yaml --image_size 512 --device cuda --dump_path ./new_controlnet-pose-sd21-diffusers
```
3. After conversion, you shall use it with package `diffuser`
```
controlnet = ControlNetModel.from_pretrained(
    "/home/ubuntu/diffusion/new_controlnet-pose-sd21-diffusers",
    torch_dtype=torch.float32,
    local_files_only=True,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet, 
    torch_dtype=torch.float32,
    #local_files_only=True,
)
```
4. Run `python sdvideo.py` after checking all arguments to control

Memo:
https://github.com/huggingface/diffusers/issues/2581 

