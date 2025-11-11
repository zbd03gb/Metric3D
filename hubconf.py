#这个文件是PyTorch Hub的标准入口点,当用户调用torch.hub.load('yvanyin/metric3d', ...)时会自动执行。

import os
import torch
try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
metric3d_dir = os.path.dirname(__file__)

# 模型配置与预训练模型下载地址
MODEL_TYPE = {
  'ConvNeXt-Tiny': {
    'cfg_file': './mono/configs/HourglassDecoder/convtiny.0.3_150.py',
    'ckpt_file': './pretrain_model/convtiny_hourglass_v1.pth',
  },
  'ConvNeXt-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
  },
  'ViT-Small': {
    'cfg_file': './mono/configs/HourglassDecoder/vit.raft5.small.py',
    'ckpt_file': './pretrain_model/metric_depth_vit_small_800k.pth',
  },
  'ViT-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
  },
  'ViT-giant2': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
  },
}



def metric3d_convnext_tiny(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  # if pretrain:
  #   model.load_state_dict(
  #     torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
  #     strict=False,
  #   )
  if pretrain:
    model.load_state_dict(
    torch.load(ckpt_file)['model_state_dict'], 
    strict=False,
    )
  return model

def metric3d_convnext_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_small(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg) 
  if pretrain:
    model.load_state_dict(
      torch.load(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_giant2(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model



if __name__ == '__main__':
  import cv2
  import numpy as np
  #### prepare data
  # rgb_file = 'data/wild_demo/david-kohler-VFRTXGw1VjU-unsplash.jpg'
  # rgb_file = 'data/wild_demo/jonathan-borba-CnthDZXCdoY-unsplash.jpg'
  rgb_file = 'data/wild_demo/randy-fath-G1yhU1Ej-9A-unsplash.jpg'
  # rgb_file = 'data/kitti_demo/rgb/0000000050.png'
  depth_file = 'data/kitti_demo/depth/0000000050.png'
  intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
  gt_depth_scale = 256.0 # 真值深度图的缩放因子 
  rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

  #### ajust input size to fit pretrained model
  # keep ratio resize
  input_size = (616, 1064) # for vit model
  # input_size = (544, 1216) # for convnext model
  h, w = rgb_origin.shape[:2]
  scale = min(input_size[0] / h, input_size[1] / w) # 保持高宽比
  rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR) 
  # 同步缩放相机内参
  intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
  # 填充
  padding = [123.675, 116.28, 103.53]
  h, w = rgb.shape[:2]
  pad_h = input_size[0] - h
  pad_w = input_size[1] - w
  pad_h_half = pad_h // 2
  pad_w_half = pad_w // 2
  rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
  # 记录填充信息为后续去除填充准备
  pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

  #### normalize
  mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
  std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
  rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
  rgb = torch.div((rgb - mean), std)
  rgb = rgb[None, :, :, :].cuda()

  ###################### canonical camera space ######################
  # 推理
  # model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
  model = metric3d_vit_small(pretrain = True)
  model.cuda().eval()
  with torch.no_grad():
    pred_depth, confidence, output_dict = model.inference({'input': rgb})


  # 去填充
  pred_depth = pred_depth.squeeze() # 去除batch维度
  pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
  
  # 调整到原始尺寸
  pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
  ###################### canonical camera space ######################

  #### de-canonical变换
  canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 为标准相机焦距
  pred_depth = pred_depth * canonical_to_real_scale # 现在深度为真实度量值
  pred_depth = torch.clamp(pred_depth, 0, 300) # 限制深度范围

  folder_path = './result'
  os.makedirs(folder_path, exist_ok=True)
  pred_depth_np = pred_depth.cpu().numpy()

  # 标准化深度图，确保深度值在 0 到 255 之间
  pred_depth_normalized = np.uint8(255 * (pred_depth_np / pred_depth_np.max()))
  print(pred_depth_normalized.shape[:2])
  print((h,w))

  # 使用 OpenCV 保存为灰度图像
  gray_path = os.path.join(folder_path, 'pred_depth_gray_5.png')
  cv2.imwrite(gray_path, pred_depth_normalized)
  # 保存为彩色深度图
  colored_depth = cv2.applyColorMap(pred_depth_normalized, cv2.COLORMAP_INFERNO)  # 可选：COLORMAP_JET / VIRIDIS / INFERNO 等
  color_path = os.path.join(folder_path, 'pred_depth_color_5.png')
  cv2.imwrite(color_path, colored_depth)
  #### you can now do anything with the metric depth 
  # such as evaluate predicted depth
  if depth_file is not None:
    gt_depth = cv2.imread(depth_file, -1)
    gt_depth = gt_depth / gt_depth_scale
    gt_depth = torch.from_numpy(gt_depth).float().cuda()
    assert gt_depth.shape == pred_depth.shape
    
    mask = (gt_depth > 1e-8)
    abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
    print('abs_rel_err:', abs_rel_err.item())

  #### normal are also available
  if 'prediction_normal' in output_dict: # only available for Metric3Dv2, i.e. vit model
    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
    # un pad and resize to some size if needed
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
    # you can now do anything with the normal
    # such as visualize pred_normal
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    cv2.imwrite('normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))

