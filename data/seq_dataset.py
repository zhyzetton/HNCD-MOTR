# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import cv2

import torchvision.transforms.functional as F

from torch.utils.data import Dataset
import torch

# Suppress FFmpeg warnings for MSMPEG4v2 codec
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
cv2.setLogLevel(0)


class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, det_db):
        self.seq_dir = seq_dir
        self.det_db = det_db
        self.use_video = False
        self.video_path = None
        self.video_cap = None
        self.seq_info = None  # For AICity22 det_db lookup

        # Check if this is AICity22 (video-based)
        video_path = os.path.join(seq_dir, "vdo.avi")
        if os.path.exists(video_path):
            self.use_video = True
            self.video_path = video_path
            # Don't create VideoCapture here - will create per worker
            # Get frame count without keeping the capture open
            temp_cap = cv2.VideoCapture(video_path)
            self.num_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_cap.release()
            self.image_paths = [f"{i+1:06d}.jpg" for i in range(self.num_frames)]  # 1-based frame names
            # Extract seq_info for det_db lookup (e.g., "AICity22/val/S05/c024")
            # seq_dir is like: .../AICity22_Track1_MTMC_Tracking/validation/S05/c024
            parts = seq_dir.replace("\\", "/").split("/")
            # Find AICity22 related parts
            for i, p in enumerate(parts):
                if "AICity22" in p:
                    # Get split (validation->val, train, test)
                    split_idx = i + 1
                    if split_idx < len(parts):
                        split = parts[split_idx]
                        if split == "validation":
                            split = "val"
                        # scenario/camera
                        if split_idx + 2 < len(parts):
                            scenario = parts[split_idx + 1]
                            camera = parts[split_idx + 2]
                            self.seq_info = f"AICity22/{split}/{scenario}/{camera}"
                    break
        elif "BDD100K" in seq_dir:
            image_paths = sorted(os.listdir(os.path.join(seq_dir)))
            image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
            self.image_paths = image_paths
        else:
            image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
            image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
            self.image_paths = image_paths

        self.image_height = 800
        self.image_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        return

    @staticmethod
    def load(path):
        image = cv2.imread(path)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _get_video_cap(self):
        """Get or create VideoCapture for current worker."""
        if self.video_cap is None:
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
        return self.video_cap

    def load_video_frame(self, frame_idx):
        """Load a frame from video file."""
        cap = self._get_video_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from video")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def process_image(self, image, info, frame_idx=None):
        proposals = []
        h, w = image.shape[:2]

        # Construct det_db key based on dataset type
        if self.use_video and self.seq_info:
            # For AICity22: key format is "AICity22/val/S05/c024/000001.txt"
            frame_name = info.replace('.jpg', '.txt')
            key = f"{self.seq_info}/{frame_name}"
        else:
            # For other datasets
            prefix = '/home/y23_zhanghaoyu/subject/datasets/'
            key = info[len(prefix):].replace('.jpg', '.txt') if len(info) > len(prefix) else ""

        if self.det_db and key in self.det_db:
            for line in self.det_db[key]:
                l, t, p_w, p_h, s = list(map(float, line.strip().split(',')))
                proposals.append([(l + p_w / 2) / w,
                                  (t + p_h / 2) / h,
                                  p_w / w,
                                  p_h / h,
                                  s])
        proposals = torch.as_tensor(proposals).reshape(-1, 5) if proposals else torch.zeros((0, 5))
        ori_image = image.copy()
        scale = self.image_height / min(h, w)
        if max(h, w) * scale > self.image_width:
            scale = self.image_width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        return image, ori_image, proposals

    def __getitem__(self, item):
        if self.use_video:
            image = self.load_video_frame(item)
            # Construct info path for det_db lookup
            info = self.image_paths[item]  # e.g., "000001.jpg"
        else:
            image = self.load(self.image_paths[item])
            info = self.image_paths[item]
        return self.process_image(image=image, info=info), info

    def __len__(self):
        return len(self.image_paths)

    def __del__(self):
        if self.video_cap is not None:
            self.video_cap.release()
