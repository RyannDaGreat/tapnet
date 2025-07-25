#!/usr/bin/env python3
"""
Simple interface for running TAPNext/TAPIR models, similar to run_cotracker.
Provides a clean two-function interface for loading models and running inference.
"""

import numpy as np
import torch
from functools import lru_cache
from einops import rearrange
import rp
import sys

#Includes this file and the tapnet module
sys.path.append(rp.get_parent_folder(__file__))

DEFAULT_MODEL_DIR = "~/.cache/tapnet"

def run_tapnet(
    video,
    *,
    device=None,
    queries=None,
    grid_size=None,
    grid_query_frame=0,
    model="tapnext",  # "tapir", "bootstapir", or "tapnext"
    model_dir=None,
    show_progress=True,
):
    """
    Runs the TAPNext/TAPIR model on a video for point tracking.
    TAPNext is a transformer-based model that can track any point in a video,
    with state-of-the-art accuracy and speed (as of Jun 2025).

    See https://github.com/google-deepmind/tapnet
    
    Args:
        video: Input video as either a file path string like './video.mp4', glob of image files like '/path/to/*frames.png', 
               list of PIL images, numpy array (T×H×W×3) with either dtype np.uint8 or floating point values between 0 and 1,
               or as a torch tensor (T×3×H×W) with values between 0 and 1.
        device: Torch device to run inference on, defaults to cuda if available
        queries: Query points of shape (N, 3) in format (t, y, x) for frame index
                 and pixel coordinates. Can be numpy array or torch tensor. Used for tracking specific points.
        grid_size: Size M for an N=M×M grid of tracking points on a frame. Must be provided
                   if queries is None, not used if queries are provided.
        grid_query_frame: Frame index(es) to start tracking from. Can be int or iterable 
                          of ints for multi-frame initialization. Only used when grid_size is not None (default: 0)
                          Not used if queries is not None.
        model: Which model to use: "tapir", "bootstapir", or "tapnext" (default: "tapnext")
        model_dir: Directory to cache downloaded models (default: "~/.cache/tapnet" from rp.git.tapnet.DEFAULT_MODEL_DIR)
        show_progress: If True, shows a progress bar during calculation.
    
    Returns:
        tuple: (pred_tracks, pred_visibility) where:
            - pred_tracks: numpy array of shape (T, N, 2) containing x,y coordinates
            - pred_visibility: numpy array of shape (T, N) indicating point visibility
    
    Track Modes:
        1. Grid tracking: Set grid_size > 0 to track M×M points from grid_query_frame
        2. Query tracking: Provide queries tensor to track specific points
        3. Dense tracking: Default with grid_size=20 if no queries provided

    Note: As of Jun 18 2025, using the MPS device on Mac yields incorrect results. If on Mac, use CPU.
    
    EXAMPLE:

        >>> video = rp.load_video(
        ...     "https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4",
        ...     use_cache=True,
        ... )
        ...
        ... # TAPNext
        ... tracks, visibility = run_tapnet(
        ...     video,
        ...     device="cuda",
        ...     grid_size=10,  # Track 10x10 grid of points
        ...     model='tapnext',
        ... )
        ...
        ... # Visualization
        ... colors = [rp.random_rgb_float_color() for _ in tracks[0]]
        ... new_video = []
        ... for frame_number in range(len(video)):
        ...     frame = video[frame_number]
        ...     track = tracks[frame_number]
        ...     visible = visibility[frame_number]
        ...     for idx, (color, (x, y)) in enumerate(zip(colors, track)):
        ...         if visible[idx]:
        ...             frame = rp.cv_draw_circle(
        ...                 frame,
        ...                 x,
        ...                 y,
        ...                 radius=5,
        ...                 color=color,
        ...                 antialias=True,
        ...                 copy=False,
        ...             )
        ...     new_video.append(frame)
        ... rp.fansi_print("SAVED " + rp.save_video_mp4(new_video), "blue cyan", "bold")

    """
    #THE ABOVE DOCSTRING SHOULD BE MIRRORED WITH rp.run_tapnet

    model_dir = model_dir or DEFAULT_MODEL_DIR

    # Validate that either grid_size or queries is provided
    if grid_size is None and queries is None:
        raise ValueError("Either grid_size or queries must be provided")
    
    if device is None:
        if hasattr(video, 'device'):
            device = video.device
        else:
            device = rp.select_torch_device(prefer_used=True, reserve=True)
    
    # Warn about MPS device issues
    if device == "mps":
        rp.fansi_print_lines(
            "run_tapnet WARNING: MPS device (on Mac) may produce incorrect tracking results.",
            "Consider using 'cpu' device for reliable outputs even though its slower.",
            "Even PYTORCH_ENABLE_MPS_FALLBACK=1 might not fix the accuracy issues.",
            style="yellow"
        )
    
    dtype = torch.float32
    
    # Get the model
    loaded_model = get_tapnet_model(model, device, dtype, model_dir)
    
    # Load video if it's a path
    if isinstance(video, str):
        video = rp.load_video(video)
    
    # Get original dimensions before conversion
    height, width = rp.get_video_dimensions(video)
    
    # Convert to torch tensor with proper format and device
    video_torch = rp.as_torch_images(video, device=device, dtype=dtype)  # Returns T×C×H×W in float [0,1]
    
    # Resize to 256x256 using rp function 
    video_resized = rp.torch_resize_images(video_torch, size=(256, 256))  # T×C×H×W
    
    # Convert to T×H×W×C format and normalize to [-1, 1]
    video_thwc = rearrange(video_resized, 't c h w -> t h w c')  # T×H×W×C
    video_normalized = video_thwc * 2 - 1  # Convert [0,1] to [-1,1]
    
    # Create query points
    if queries is not None:
        queries_unscaled = torch.tensor(queries, dtype=dtype, device=device)
    else:
        # Create grid points (grid_size is guaranteed to be not None due to validation above)
        queries_unscaled = _create_grid_queries(grid_size, grid_query_frame, height, width, device, dtype)
    
    # Scale x and y coordinates to 256x256 for model input
    queries_scaled = queries_unscaled.clone()
    queries_scaled[:, 1] *= 256 / width  # x
    queries_scaled[:, 2] *= 256 / height # y

    queries_scaled_yx = queries_scaled[:, [0, 2, 1]]  # XY -> YX

    rp.validate_tensor_shapes(
        queries_unscaled  = 'torch: N TXY',
        queries_scaled    = 'torch: N TXY',
        queries_scaled_yx = 'torch: N TXY',
        video_normalized  = 'torch: T H W C',
        video_resized     = 'torch: T C H W',
        TXY=3,
    )
    
    # Run inference
    with torch.no_grad():
        outputs = loaded_model(
            video_normalized[None],
            queries_scaled_yx[None],
            show_progress=show_progress,
        )
    
    # Extract tracks and visibility
    pred_tracks = outputs['tracks'][0]  # N×T×2 (XY)
    pred_occlusion = outputs['occlusion'][0]  # N×T
    
    # Convert occlusion to visibility
    pred_visibility = torch.sigmoid(pred_occlusion) < 0.5  # Lower occlusion = visible
    
    # Rearrange to T×N×2 format to match run_cotracker
    pred_tracks = rearrange(pred_tracks, 'n t d -> t n d')  # T×N×2 (XY)
    pred_visibility = rearrange(pred_visibility, 'n t -> t n')  # T×N
    
    # Scale tracks back to original resolution
    pred_tracks[..., 0] *= width / 256   # x
    pred_tracks[..., 1] *= height / 256  # y
    
    # Convert to CPU numpy arrays for compatibility
    pred_tracks = pred_tracks
    pred_visibility = pred_visibility
    
    # Validate output tensor shapes
    rp.validate_tensor_shapes(
        pred_tracks="torch: T N XY",
        pred_visibility="torch: T N",
        video="T H W C",  # Original input video
        verbose=False,
        XY=2,
    )
    
    return pred_tracks, pred_visibility


@lru_cache(3)
def _create_grid_queries(grid_size, grid_query_frame, height, width, device, dtype):
    """Create grid query points with proper spacing-based margins.
    
    Args:
        grid_size: Size of the grid (M for M×M grid)
        grid_query_frame: Frame index(es) to start tracking from. Can be int or iterable of ints.
        height: Height of the video
        width: Width of the video  
        device: Torch device
        dtype: Torch dtype
    """
    # Handle multiple frames - normalize to list
    if isinstance(grid_query_frame, (list, tuple)):
        frame_list = list(grid_query_frame)
    elif isinstance(grid_query_frame, int):
        frame_list = [grid_query_frame]
    else:
        raise ValueError(f"grid_query_frame must be int, list, or tuple, got {type(grid_query_frame)}")
    
    # Margin is half the spacing between grid points
    x_spacing = width  / (grid_size + 1)
    y_spacing = height / (grid_size + 1)
    x_margin = x_spacing / 2
    y_margin = y_spacing / 2
    x_start, x_end = x_margin, width  - x_margin
    y_start, y_end = y_margin, height - y_margin
    
    x_coords = torch.linspace(x_start, x_end, grid_size)
    y_coords = torch.linspace(y_start, y_end, grid_size)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    
    queries_list = []
    for frame_idx in frame_list:
        for x, y in zip(grid_x.flatten(), grid_y.flatten()):
            queries_list.append([frame_idx, x.item(), y.item()])
    
    queries_tensor = torch.tensor(queries_list, dtype=dtype).to(device) # N x 3

    return queries_tensor

    
# Model checkpoint URLs
checkpoint_urls = {
    "tapir": "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt",
    "bootstapir": "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",
    "tapnext": "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",  # Using BootsTAPIR as placeholder
}

@lru_cache()
def get_tapnet_model(model="tapnext", device=None, dtype=None, model_dir=None, download_only=False):
    """
    Loads and caches the TAPNext/TAPIR model.
    
    Args:
        model: Which model to load: "tapir", "bootstapir", or "tapnext"
        device: The torch device to load the model onto
        dtype: The torch dtype to use
        model_dir: The folder where we cache checkpoints
        download_only: If true, just checks that its downloaded and returns the checkpoint.pt path string
    
    Returns:
        The loaded TAPNext/TAPIR model
    
    Downloads are cached in ~/.cache/tapnet/
    """
    if not model in checkpoint_urls:
        raise ValueError(f"rp.r.git.tapnet.get_tapnet_model: {repr(model)} is not a valid model; please choose from {list(checkpoint_urls)}")

    from tapnet.torch import tapir_model

    model_dir = model_dir or DEFAULT_MODEL_DIR
    
    # Create cache directory
    model_dir = rp.get_absolute_path(model_dir)
    rp.make_directory(model_dir)
    
    # Download checkpoint if needed
    checkpoint_name = f"{model}_checkpoint.pt"
    checkpoint_path = rp.path_join(model_dir, checkpoint_name)
    
    if not rp.path_exists(checkpoint_path):
        print(f"Downloading {model} checkpoint...")
        url = checkpoint_urls.get(model, checkpoint_urls[model])
        rp.download_url(url, checkpoint_path, show_progress=True)
        print(f"Downloaded {model} checkpoint to {checkpoint_path}")

    if download_only:
        return checkpoint_path
    
    # Initialize model
    print(f"Loading {model} model from {checkpoint_path}...")
    model = tapir_model.TAPIR(pyramid_level=1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and dtype
    if dtype is not None:
        model = model.to(dtype)
    if device is not None:
        model = model.to(device)
    
    model = model.eval()
    
    return model


if __name__ == "__main__":
    # Demo usage
    print("Running TAPNext demo...")
    
    # Load video
    video = rp.load_video(
        "https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4",
        use_cache=True,
    )
    
    # Run TAPNext
    tracks, visibility = run_tapnet(
        video,
        device="cuda" if torch.cuda.is_available() else "cpu",
        grid_size=10,  # Track 10x10 grid of points
    )
    
    # Visualization
    colors = [rp.random_rgb_float_color() for _ in tracks[0]]
    new_video = []
    for frame_number in range(len(video)):
        frame = video[frame_number].copy()  # Copy once at the beginning
        track = tracks[frame_number]
        visible = visibility[frame_number]
        for idx, (color, (x, y)) in enumerate(zip(colors, track)):
            if visible[idx]:
                frame = rp.cv_draw_circle(
                    frame,
                    x,
                    y,
                    radius=5,
                    color=color,
                    antialias=True,
                    copy=False,  # Fast mutation since we already copied
                )
        new_video.append(frame)
    
    output_path = rp.save_video_mp4(new_video, "output_tapnet.mp4")
    rp.fansi_print(f"SAVED {output_path}", "blue cyan", "bold")
