"""
ComfyUI Mask Blend Node
图像序列Mask渐变过渡节点
"""

import torch
import numpy as np

class MaskGradientNode:
    """
    创建从黑色到白色的mask渐变过渡效果
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 图像序列 [B, H, W, C]
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "end_frame": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "generate_gradient_mask"
    CATEGORY = "mask/transitions"
    
    def generate_gradient_mask(self, images, start_frame, end_frame):
        """
        生成渐变mask序列
        
        Args:
            images: 输入图像序列，torch.Tensor [B, H, W, C]
            start_frame: 渐变开始帧索引
            end_frame: 渐变结束帧索引
            
        Returns:
            images: 原始图像序列
            masks: 生成的mask序列 [B, H, W]
        """
        # 获取图像序列的形状
        batch_size, height, width, channels = images.shape
        
        # 创建mask序列，初始化为全黑（0）
        masks = torch.zeros((batch_size, height, width), dtype=torch.float32, device=images.device)
        
        # 确保start_frame和end_frame在有效范围内
        start_frame = max(0, min(start_frame, batch_size - 1))
        end_frame = max(0, min(end_frame, batch_size - 1))
        
        # 如果start_frame大于end_frame，交换它们
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        
        # 在指定范围内创建渐变
        if start_frame == end_frame:
            # 如果起始帧和结束帧相同，该帧设置为白色
            if start_frame < batch_size:
                masks[start_frame] = 1.0
        else:
            # 创建从黑到白的渐变
            transition_frames = end_frame - start_frame
            for i in range(start_frame, end_frame + 1):
                if i < batch_size:
                    # 计算当前帧的渐变值（0.0到1.0）
                    gradient_value = (i - start_frame) / transition_frames
                    masks[i] = gradient_value
        
        return (images, masks)


class FrameSliceNode:
    """
    从图像序列中提取指定范围的帧
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 图像序列 [B, H, W, C]
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "end_frame": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "slice_frames"
    CATEGORY = "image/sequence"
    
    def slice_frames(self, images, start_frame, end_frame):
        """
        提取指定范围的图像帧
        
        Args:
            images: 输入图像序列，torch.Tensor [B, H, W, C]
            start_frame: 起始帧索引（包含）
            end_frame: 结束帧索引（包含）
            
        Returns:
            images: 提取的图像序列切片
        """
        # 获取图像序列的批次大小
        batch_size = images.shape[0]
        
        # 确保start_frame和end_frame在有效范围内
        start_frame = max(0, min(start_frame, batch_size - 1))
        end_frame = max(0, min(end_frame, batch_size - 1))
        
        # 如果start_frame大于end_frame，交换它们
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        
        # 提取指定范围的帧（end_frame+1是因为切片不包含结束索引）
        sliced_images = images[start_frame:end_frame + 1]
        
        return (sliced_images,)


class SequenceOverlayNode:
    """
    将第一段图像序列按照mask叠加到第二段图像序列上
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_images": ("IMAGE",),  # 第一段图像序列 [B1, H, W, C]
                "second_images": ("IMAGE",),  # 第二段图像序列 [B2, H, W, C]
                "masks": ("MASK",),  # mask序列 [B1, H, W]
                "overlay_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "overlay_sequences"
    CATEGORY = "image/sequence"
    
    def overlay_sequences(self, first_images, second_images, masks, overlay_start):
        """
        将第一段图像序列按照mask叠加到第二段图像序列上
        
        Args:
            first_images: 第一段图像序列，torch.Tensor [B1, H, W, C]
            second_images: 第二段图像序列，torch.Tensor [B2, H, W, C]
            masks: mask序列，torch.Tensor [B1, H, W]，值范围0.0-1.0
            overlay_start: 第一段中从哪一帧开始与第二段对齐（前面的帧单独显示）
            
        Returns:
            images: 叠加后的图像序列
        """
        # 获取序列信息
        first_len = first_images.shape[0]
        second_len = second_images.shape[0]
        height, width, channels = first_images.shape[1], first_images.shape[2], first_images.shape[3]
        
        # 确保overlay_start在有效范围内
        overlay_start = max(0, min(overlay_start, first_len))
        
        # 计算输出序列的长度
        # 前overlay_start帧是第一段单独显示
        # 之后是叠加部分和剩余部分
        first_remaining = first_len - overlay_start  # 第一段在overlay_start之后还剩多少帧
        output_len = overlay_start + max(first_remaining, second_len)
        
        # 创建输出序列，初始化为黑色
        output_images = torch.zeros(
            (output_len, height, width, channels),
            dtype=first_images.dtype,
            device=first_images.device
        )
        
        # 第一部分：前overlay_start帧，只显示第一段
        if overlay_start > 0:
            output_images[:overlay_start] = first_images[:overlay_start]
        
        # 第二部分：叠加部分（从overlay_start开始）
        # 计算叠加的帧数
        overlay_frames = min(first_remaining, second_len)
        
        for i in range(overlay_frames):
            first_idx = overlay_start + i  # 第一段的帧索引
            second_idx = i  # 第二段的帧索引
            output_idx = overlay_start + i  # 输出的帧索引
            
            # 获取当前帧的mask [H, W]
            mask = masks[first_idx]
            
            # 将mask扩展到3个通道 [H, W, C]
            mask_3ch = mask.unsqueeze(-1).expand(-1, -1, channels)
            
            # Alpha混合：output = first * mask + second * (1 - mask)
            foreground = first_images[first_idx]
            background = second_images[second_idx]
            
            output_images[output_idx] = foreground * mask_3ch + background * (1 - mask_3ch)
        
        # 第三部分：处理剩余帧
        remaining_start = overlay_start + overlay_frames
        
        if first_remaining > second_len:
            # 第一段更长，继续显示第一段剩余部分
            remaining_first_frames = first_remaining - second_len
            for i in range(remaining_first_frames):
                first_idx = overlay_start + overlay_frames + i
                output_idx = remaining_start + i
                
                # 获取mask
                mask = masks[first_idx]
                mask_3ch = mask.unsqueeze(-1).expand(-1, -1, channels)
                
                # 底层是黑色，只显示第一段
                foreground = first_images[first_idx]
                output_images[output_idx] = foreground * mask_3ch
                
        elif second_len > first_remaining:
            # 第二段更长，继续显示第二段剩余部分
            remaining_second_frames = second_len - first_remaining
            output_images[remaining_start:remaining_start + remaining_second_frames] = \
                second_images[first_remaining:second_len]
        
        return (output_images,)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskGradientNode": MaskGradientNode,
    "FrameSliceNode": FrameSliceNode,
    "SequenceOverlayNode": SequenceOverlayNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskGradientNode": "Mask Gradient (渐变过渡)",
    "FrameSliceNode": "Frame Slice (帧切片)",
    "SequenceOverlayNode": "Sequence Overlay (序列叠加)",
}

