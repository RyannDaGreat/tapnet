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

sys.path.append(rp.get_parent_folder(__file__))

def run_tapnext(
    video,
    *,
    device=None,
    queries=None,
    grid_size=None,
    grid_query_frame=0,
    model="tapnext",  # "tapir", "bootstapir", or "tapnext"
    model_dir="~/.cache/tapnet",
):
    """
    Runs the TAPNext/TAPIR model on a video for point tracking.
    TAPNext is a transformer-based model that can track any point in a video,
    with state-of-the-art accuracy and speed.
    
    Args:
        video: Input video as either a file path, numpy array (T×H×W×C), or torch tensor (T×C×H×W).
        device: Torch device to run inference on, defaults to cuda if available
        queries: Query points of shape (N, 3) in format (t, x, y) for frame index
                and pixel coordinates. Can be numpy array or torch tensor. Used for tracking specific points.
        grid_size: Size M for an N=M×M grid of tracking points on a frame. Must be provided
                  if queries is None.
        grid_query_frame: Frame index(es) to start tracking from. Can be int or iterable 
                         of ints for multi-frame initialization. Only used when grid_size is not None (default: 0)
        model: Which model to use: "tapir", "bootstapir", or "tapnext"
        model_dir: Directory to cache downloaded models (default: "~/.cache/tapnet")
    
    Returns:
        tuple: (pred_tracks, pred_visibility) where:
            - pred_tracks: torch tensor of shape (T, N, 2) containing x,y coordinates
            - pred_visibility: torch tensor of shape (T, N) indicating point visibility
    
    Track Modes:
        1. Grid tracking: Set grid_size > 0 to track M×M points from grid_query_frame
        2. Query tracking: Provide queries tensor to track specific points
        3. Dense tracking: Default with grid_size=20 if no queries provided
    
    EXAMPLE:
        >>> video = rp.load_video(
        ...     "https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4",
        ...     use_cache=True,
        ... )
        ...
        ... # TAPNext
        ... tracks, visibility = run_tapnext(
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
            "run_tapnext WARNING: MPS device may produce incorrect tracking results.",
            "Consider using 'cuda' or 'cpu' device for reliable outputs.",
            "Even PYTORCH_ENABLE_MPS_FALLBACK=1 will not fix the accuracy issues.",
            style="yellow"
        )
    
    dtype = torch.float32
    
    # Get the model
    loaded_model = _get_tapnext_model(model, device, dtype, model_dir)
    
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
        outputs = loaded_model(video_normalized[None], queries_scaled_yx[None])
    
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


@lru_cache(maxsize=3)
def _get_tapnext_model(model="tapnext", device=None, dtype=None, model_dir="~/.cache/tapnet"):
    """
    Loads and caches the TAPNext/TAPIR model.
    
    Args:
        model: Which model to load: "tapir", "bootstapir", or "tapnext"
        device: The torch device to load the model onto
        dtype: The torch dtype to use
    
    Returns:
        The loaded TAPNext/TAPIR model
    
    Downloads are cached in ~/.cache/tapnet/
    """
    from tapnet.torch import tapir_model
    
    # Model checkpoint URLs
    checkpoint_urls = {
        "tapir": "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt",
        "bootstapir": "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",
        "tapnext": "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",  # Using BootsTAPIR as placeholder
    }
    
    # Create cache directory
    model_dir = rp.get_absolute_path(model_dir)
    rp.make_directory(model_dir)
    
    # Download checkpoint if needed
    checkpoint_name = f"{model}_checkpoint.pt"
    checkpoint_path = rp.path_join(model_dir, checkpoint_name)
    
    if not rp.path_exists(checkpoint_path):
        print(f"Downloading {model} checkpoint...")
        url = checkpoint_urls.get(model, checkpoint_urls["bootstapir"])
        rp.download_url(url, checkpoint_path, show_progress=True)
        print(f"Downloaded {model} checkpoint to {checkpoint_path}")
    
    # Initialize model
    print(f"Loading {model} model...")
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
    tracks, visibility = run_tapnext(
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
    
    output_path = rp.save_video_mp4(new_video, "output_tapnext.mp4")
    rp.fansi_print(f"SAVED {output_path}", "blue cyan", "bold")
