import torch
import mmengine

ckpt = torch.load("/home/chenxu/code/mmdetection3d/work_dirs/bevfusion_yx_kl/epoch_5.pth", map_location="cpu")
state_dict = ckpt["state_dict"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

torch.save(state_dict, "bevfusion_epoch5_state_dict.pth")
