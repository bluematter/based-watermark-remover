# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List
import subprocess
import tempfile
import os

from cog import BasePredictor, Input, Path
import inference_propainter as propainter


def get_video_frame_count(video_path):
    """Get the number of frames in a video using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames',
        '-of', 'csv=p=0', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


def get_video_fps(video_path):
    """Get the FPS of a video using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'csv=p=0', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # r_frame_rate is in format "num/denom"
    num, denom = result.stdout.strip().split('/')
    return float(num) / float(denom)


def get_video_dimensions(video_path):
    """Get width and height of a video using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    width, height = result.stdout.strip().split(',')
    return int(width), int(height)


def download_mask_for_aspect_ratio(video_path, temp_dir):
    """Download the appropriate mask based on video aspect ratio."""
    width, height = get_video_dimensions(video_path)
    aspect_ratio = width / height

    # Determine if video is landscape (16:9) or portrait (9:16)
    # 16:9 = 1.778, 9:16 = 0.5625
    # Use threshold of 1.0 to distinguish
    if aspect_ratio > 1.0:
        # Landscape - use 16:9 mask
        mask_url = "https://media.basedlabs.ai/mask_16-9_static_450.mp4"
        print(f"Detected landscape video ({width}x{height}, ratio: {aspect_ratio:.2f}), using 16:9 mask")
    else:
        # Portrait - use 9:16 mask
        mask_url = "https://media.basedlabs.ai/mask_9-16_static_450.mp4"
        print(f"Detected portrait video ({width}x{height}, ratio: {aspect_ratio:.2f}), using 9:16 mask")

    # Download mask
    mask_filename = os.path.join(temp_dir, os.path.basename(mask_url))
    cmd = ['curl', '-L', '-o', mask_filename, mask_url]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Downloaded mask to {mask_filename}")

    return mask_filename


def preprocess_mask_video(mask_path, target_frames, temp_dir):
    """Preprocess mask video to match target frame count using ffmpeg."""
    mask_frames = get_video_frame_count(mask_path)

    if mask_frames == target_frames:
        print(f"Mask already has {target_frames} frames, no preprocessing needed")
        return mask_path

    output_path = os.path.join(temp_dir, f"preprocessed_mask{os.path.splitext(mask_path)[1]}")

    print(f"Preprocessing mask: {mask_frames} frames -> {target_frames} frames")

    if mask_frames > target_frames:
        # Trim excess frames using select filter
        cmd = [
            'ffmpeg', '-i', str(mask_path), '-vf', f'select=lt(n\\,{target_frames})',
            '-vsync', '0', '-y', output_path
        ]
    else:
        # Extend by repeating last frame
        mask_fps = get_video_fps(mask_path)
        extra_frames = target_frames - mask_frames
        pad_duration = extra_frames / mask_fps
        cmd = [
            'ffmpeg', '-i', str(mask_path), '-vf', f'tpad=stop_mode=clone:stop_duration={pad_duration}',
            '-y', output_path
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Saved preprocessed mask to {output_path}")

    return output_path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device, self.fix_raft, self.fix_flow_complete, self.model = propainter.setup()

    def predict(
        self,
        video: Path = Input(description="Input video"),
        mask: Path = Input(description="Mask for video inpainting. Can be a static image (jpg, png) or a video (avi, mp4). If not provided, will auto-select based on video aspect ratio (16:9 landscape or 9:16 portrait).", default=None),
        return_input_video: bool = Input(description="Return the input video in the output.", default=False),
        resize_ratio: float = Input(description="Resize scale for processing video.", default=1.0),
        height: int = Input(description="Height of the processing video.", default=-1),
        width: int = Input(description="Width of the processing video.", default=-1),
        mask_dilation: int = Input(description="Mask dilation for video and flow masking.", default=4),
        ref_stride: int = Input(description="Stride of global reference frames.", default=10),
        neighbor_length: int = Input(description="Length of local neighboring frames.", default=10),
        subvideo_length: int = Input(description="Length of sub-video for long video inference.", default=80),
        raft_iter: int = Input(description="Iterations for RAFT inference.", default=20),
        mode: str = Input(description="Modes: video inpainting / video outpainting. If you want to do video inpainting, you need a mask. For video outpainting, you need to set scale_h and scale_w, and mask is ignored.", choices=['video_inpainting', 'video_outpainting'], default='video_inpainting'),
        scale_h: float = Input(description="Outpainting scale of height for video_outpainting mode.", default=1.0),
        scale_w: float = Input(description="Outpainting scale of width for video_outpainting mode.", default=1.0),
        save_fps: int = Input(description="Frames per second.", default=24),
        fp16: bool = Input(description="Use fp16 (half precision) during inference. Default: fp32 (single precision).", default=False)
        
    ) -> List[Path]:
        """Run a single prediction on the model"""
        output = "results"
        save_frames=False

        if mode in ['video_inpainting']:
            # Mask is now optional - will auto-detect based on aspect ratio if not provided
            if mask and mask.suffix.lower() not in ['.mp4','.avi','.png','.jpg']:
                raise ValueError("ProPainter via cog only supports static masks as .jpg or .png, or video masks as .avi and .mp4.")

        if mode in ['video_outpainting']:
            if not scale_w or not scale_h:
                raise ValueError("Video outpainting needs scale_h and scale_w as input parameters.")

        # Auto-detect and download mask based on aspect ratio if not provided
        with tempfile.TemporaryDirectory() as temp_dir:
            if mode == 'video_inpainting':
                if not mask:
                    # Auto-select mask based on video aspect ratio
                    print("No mask provided, auto-detecting based on video aspect ratio...")
                    mask_path = download_mask_for_aspect_ratio(video, temp_dir)
                else:
                    mask_path = str(mask)

                # Preprocess mask to match input video frame count
                if mask_path.endswith(('.mp4', '.avi', '.MP4', '.AVI')):
                    video_frames = get_video_frame_count(video)
                    mask_path = preprocess_mask_video(mask_path, video_frames, temp_dir)

                in_video, out_video = propainter.infer(
                    self.device, self.fix_raft, self.fix_flow_complete, self.model,
                    str(video), mask_path, output, resize_ratio, height, width,
                    mask_dilation, ref_stride, neighbor_length, subvideo_length, raft_iter,
                    mode, scale_h, scale_w, save_fps, save_frames, fp16
                )
            else:
                # video_outpainting mode
                in_video, out_video = propainter.infer(
                    self.device, self.fix_raft, self.fix_flow_complete, self.model,
                    str(video), str(mask) if mask else None, output, resize_ratio, height, width,
                    mask_dilation, ref_stride, neighbor_length, subvideo_length, raft_iter,
                    mode, scale_h, scale_w, save_fps, save_frames, fp16
                )

        if return_input_video:
            return [Path(in_video), Path(out_video)]
        return [Path(out_video)]
