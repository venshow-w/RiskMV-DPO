import os
from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional
import logging

import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmengine.config import ConfigDict

from magicdrivedit.utils.misc import format_numel_str
from magicdrivedit.registry import DATASETS, build_module
from ..mmdet_plugin.datasets import NuScenesDataset
from .nuscenes_t_dataset import NuScenesTDataset, collate_fn_single_clip, transform_bbox, obtain_next2top
from .utils import IMG_FPS
from .nuscenes_variable import NuScenesVariableDataset


@DATASETS.register_module()
class NuScenesVariableDepthDataset(NuScenesVariableDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length: list[int] = None,
        start_on_keyframe=True,
        next2topv2=True,
        trans_box2top=False,
        base_fps=12,
        fps: list[list[int]] = None,
        repeat_times: list[int] = None,
        img_collate_param={},
        micro_frame_size=None,
        balance_keywords=None,
        drop_ori_imgs=False,
        drop_first_frames=False,
        **kwargs,
    ) -> None:
        super().__init__(ann_file=ann_file, 
        pipeline=pipeline, 
        dataset_root=dataset_root, 
        object_classes=object_classes, 
        map_classes=map_classes, 
        load_interval=load_interval, 
        with_velocity=with_velocity, 
        modality=modality, 
        box_type_3d=box_type_3d, 
        filter_empty_gt=filter_empty_gt, 
        test_mode=test_mode, 
        eval_version=eval_version, 
        use_valid_flag=use_valid_flag, 
        force_all_boxes=force_all_boxes, 
        video_length=video_length,  
        start_on_keyframe=start_on_keyframe, 
        next2topv2=next2topv2, 
        trans_box2top=trans_box2top,
        base_fps=base_fps, 
        fps=fps,
        repeat_times=repeat_times, 
        img_collate_param=img_collate_param, 
        micro_frame_size=micro_frame_size, 
        balance_keywords=balance_keywords, 
        drop_ori_imgs=drop_ori_imgs, 
        drop_first_frames=drop_first_frames
        )

        self.depth_root = kwargs.pop('depth_root', None)
        self.depth_format = kwargs.pop('depth_format', '.pth')
        if self.depth_root is None:
            logging.warning("Depth root is not provided in cfg, depth maps will not be loaded.")
    
    def __len__(self):
        return len(self.clip_infos)
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = OrderedDict()
        for idx, video_length in enumerate(self.video_lengths):
            if self.repeat_times is not None:
                repeat_times = self.repeat_times[idx]
            else:
                repeat_times = 1
            self.clip_infos[video_length] = self.build_clips(
                data_infos, data['scene_tokens'], video_length, repeat_times)
        return data_infos

    
    def load_depth_map(self, depth_path):
        """
        根据帧信息（frame_info）加载对应的深度图
        """
        if not os.path.exists(depth_path):
            logging.error(f"Depth map not found: {depth_path}")
            return None
        
        # 加载深度图
        data = np.load(depth_path) # data.files: ['image', 'depth', 'conf', 'intrinsics']
        depth = data['depth']
        return depth
    
    
    def load_frames(self, frames):
        if None in frames:
            return None
        examples = []
        first_frame_boxes = None
        for frame in frames:
            self.pre_pipeline(frame)
            example = self.pipeline(frame)
            cameras2ego = frame['camera2ego'] #[(4,4)]
            ego2global = frame['ego2global'] # (4,4)
            cameras2global = [ego2global @ camera2ego for camera2ego in cameras2ego]
            global2cameras = [np.linalg.inv(camera2global) for camera2global in cameras2global]
            example['global2cameras'] = np.array(global2cameras)
            
            filenames = frame['filename']
            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            if self.trans_box2top:
                if first_frame_boxes is None:
                    first_frame_boxes = {
                        'gt_bboxes_3d': example['gt_bboxes_3d'],
                        'gt_labels_3d': example['gt_labels_3d'],
                    }
                else:
                    this_frame_boxes = transform_bbox(
                        first_frame_boxes, frame['next2top'])
                    example['gt_bboxes_3d'] = this_frame_boxes['gt_bboxes_3d']
                    example['gt_labels_3d'] = this_frame_boxes['gt_labels_3d']
            examples.append(example)
        if self.del_box_ratio > 0 or self.allow_class is not None or self.drop_nearest_car > 0:
            # will change in-place
            self.rand_del_box(
                examples, self.del_box_ratio, self.allow_class, self.drop_nearest_car)
        ret_dicts = collate_fn_single_clip(examples, **self.img_collate_param)
        
        if self.img_collate_param.get("return_raw_data", False):
            return ret_dicts
        ret_dicts['height'] = ret_dicts['pixel_values'].shape[-2]
        ret_dicts['width'] = ret_dicts['pixel_values'].shape[-1]
        if self.drop_ori_imgs:
            ret_dicts["pixel_values_shape"] = torch.IntTensor(
                list(ret_dicts['pixel_values'].shape))
            ret_dicts.pop("pixel_values")
        ret_dicts['first_frames'] = ret_dicts['pixel_values'][:2,] ########################## fist 2 frames for dynamic dggt
        # print('==============++++++++++++++===============',ret_dicts['pixel_values'].shape)
        # if not self.drop_first_frames:
        #     ret_dicts['first_frames'] = ret_dicts['pixel_values'][:1,] # (17,6,3,848,1600)
        # else:
        #     ret_dicts['first_frames'] = None
        
        return ret_dicts
    
    
    def get_data_info(self, idx, num_frames, interval):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[num_frames][idx][0::interval]
        frames = self.load_clip(clip)
        return frames

    def prepare_train_data(self, index):
        idx, real_t, fps = self.parse_index(index)
        if isinstance(real_t, str) or real_t > 1:
            assert fps <= self.base_fps
            interval = self.base_fps // fps
        else:
            interval = 1
        frames = self.get_data_info(idx, real_t, interval=interval)
        real_t = len(frames)  # NOTE: we have load interval, real_t may change
        ret_dicts = self.load_frames(frames)
      
        if ret_dicts is None:
            return None
        ret_dicts['fps'] = IMG_FPS if real_t == 1 else fps
        ret_dicts['num_frames'] = real_t
        return ret_dicts
        
