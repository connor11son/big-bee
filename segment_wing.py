"""
Wing segmentation pipeline.

Three-stage pipeline:
  1. crop_wing        – Detect and crop the wing from a raw image using SAM3.
  2. segment_wing     – Generate fine-grained segmentation masks on a cropped wing using SAM.
  3. save_masks_tiff  – Write masks to an OME-TIFF file (one channel per mask).

Every function accepts a `device` argument ("cpu", "cuda", or "cuda:0", etc.)
so callers can pin execution to a specific backend.
"""

import gc
import os

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Preset 30 distinct colors (RGB, 0-1 range) for optional visualisation
# ---------------------------------------------------------------------------
MASK_COLORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 0.0],
    [0.5, 0.0, 1.0],
    [0.0, 1.0, 0.5],
    [1.0, 0.0, 0.5],
    [0.5, 1.0, 0.0],
    [0.0, 0.5, 1.0],
    [0.8, 0.4, 0.0],
    [0.4, 0.0, 0.8],
    [0.0, 0.8, 0.4],
    [0.8, 0.0, 0.4],
    [0.4, 0.8, 0.0],
    [0.0, 0.4, 0.8],
    [1.0, 0.7, 0.7],
    [0.7, 1.0, 0.7],
    [0.7, 0.7, 1.0],
    [1.0, 1.0, 0.7],
    [1.0, 0.7, 1.0],
    [0.7, 1.0, 1.0],
    [0.6, 0.3, 0.0],
    [0.3, 0.0, 0.6],
    [0.0, 0.6, 0.3],
    [0.6, 0.0, 0.3],
    [0.3, 0.6, 0.0],
    [0.0, 0.3, 0.6],
]


# ── helpers ────────────────────────────────────────────────────────────────
def _resolve_device(device: str | None = None) -> str:
    """Return the requested device, falling back to CUDA when available."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unload(*objects):
    """Delete objects and reclaim GPU / CPU memory."""
    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ── 1. crop ────────────────────────────────────────────────────────────────
def crop_wing(
    image_path: str,
    text_prompt: str = "wing",
    threshold: float = 0.5,
    device: str | None = None,
) -> tuple[Image.Image, np.ndarray] | tuple[None, None]:
    """Detect and crop the largest wing region from *image_path* using SAM3.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    text_prompt : str
        Text prompt fed to the SAM3 processor (default ``"wing"``).
    threshold : float
        Mask / detection threshold (default ``0.5``).
    device : str | None
        PyTorch device string.  ``None`` → auto-detect.

    Returns
    -------
    tuple[Image.Image, np.ndarray] | tuple[None, None]
        ``(cropped_pil_image, cropped_mask_bool_array)`` on success,
        or ``(None, None)`` if no masks were found.
    """
    from transformers import Sam3Model, Sam3Processor

    device = _resolve_device(device)
    print(f"[crop_wing] Loading SAM3 model on {device} …")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    try:
        image = Image.open(image_path)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"]
        print(
            f"[crop_wing] Found {len(masks)} object(s) in {os.path.basename(image_path)}"
        )

        if len(masks) == 0:
            print("[crop_wing] No masks found – skipping crop.")
            return None, None

        # Pick the largest mask by pixel area
        largest_idx = max(range(len(masks)), key=lambda i: masks[i].sum().item())
        largest_area = masks[largest_idx].sum().item()
        print(f"[crop_wing] Largest mask is #{largest_idx} (area {largest_area})")

        largest_mask = masks[largest_idx].cpu().numpy()
        y_coords, x_coords = np.where(largest_mask)

        if len(y_coords) == 0:
            return None, None

        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
        cropped_mask = largest_mask[y_min : y_max + 1, x_min : x_max + 1]

        print(f"[crop_wing] Cropped bbox: ({x_min},{y_min}) → ({x_max},{y_max})")
        return cropped_image, cropped_mask

    finally:
        _unload(model, processor)


# ── 2. segment ─────────────────────────────────────────────────────────────
def segment_wing(
    cropped_image: Image.Image,
    wing_mask: np.ndarray,
    overlap_threshold: float = 0.7,
    device: str | None = None,
) -> list[np.ndarray]:
    """Run SAM auto-mask generation on a cropped wing image.

    Masks that do not sufficiently overlap the *wing_mask* are discarded, and
    larger masks that mostly duplicate a smaller, already-kept mask are removed.

    Parameters
    ----------
    cropped_image : PIL.Image.Image
        The cropped wing image (output of :func:`crop_wing`).
    wing_mask : np.ndarray
        Boolean / binary mask of the wing region (same spatial size as
        *cropped_image*).
    overlap_threshold : float
        Minimum fraction of a candidate mask that must fall inside
        *wing_mask* for it to be kept (default ``0.7``).
    device : str | None
        PyTorch device string.  ``None`` → auto-detect.

    Returns
    -------
    list[np.ndarray]
        Deduplicated segmentation masks sorted by area (smallest first).
    """
    from transformers import pipeline

    device = _resolve_device(device)
    print(f"[segment_wing] Loading SAM (facebook/sam-vit-huge) on {device} …")
    generator = pipeline(
        "mask-generation", model="facebook/sam-vit-huge", device=device
    )

    try:
        results = generator(cropped_image)
        all_masks = results["masks"]
        print(f"[segment_wing] SAM returned {len(all_masks)} masks total.")

        # ── keep only masks that overlap the wing ──────────────────────
        wing_filtered: list[tuple[np.ndarray, int]] = []
        for mask in all_masks:
            mask_np = np.asarray(mask)
            mask_area = int(mask_np.sum())
            if mask_area == 0:
                continue
            overlap = np.logical_and(mask_np, wing_mask).sum() / mask_area
            if overlap >= overlap_threshold:
                wing_filtered.append((mask_np, mask_area))

        print(
            f"[segment_wing] {len(wing_filtered)} masks pass wing-overlap "
            f"threshold ({overlap_threshold})."
        )

        # ── deduplicate: prefer smaller masks ──────────────────────────
        wing_filtered.sort(key=lambda x: x[1])  # ascending area
        final_masks: list[np.ndarray] = []

        for mask_np, _ in wing_filtered:
            duplicate = False
            for kept in final_masks:
                kept_area = int(kept.sum())
                if kept_area > 0:
                    overlap_with_kept = np.logical_and(mask_np, kept).sum() / kept_area
                    if overlap_with_kept > 0.5:
                        duplicate = True
                        break
            if not duplicate:
                final_masks.append(mask_np)

        print(f"[segment_wing] {len(final_masks)} masks after deduplication.")
        return final_masks

    finally:
        _unload(generator)


# ── 3. save ────────────────────────────────────────────────────────────────
def save_masks_tiff(
    masks: list[np.ndarray],
    output_path: str,
) -> str:
    """Save a list of binary masks as an OME-TIFF (one channel per mask).

    Parameters
    ----------
    masks : list[np.ndarray]
        Segmentation masks (each 2-D, boolean or uint8).
    output_path : str
        Destination file path.  The extension is normalised to
        ``.ome.tiff`` automatically.

    Returns
    -------
    str
        The (possibly corrected) path the file was actually written to.

    Raises
    ------
    ValueError
        If *masks* is empty.
    ImportError
        If *tifffile* is not installed.
    """
    if not masks:
        raise ValueError("No masks to save.")

    import tifffile

    first = np.asarray(masks[0]).squeeze()
    height, width = first.shape
    num = len(masks)

    # Build a (C, Y, X) uint8 array – 255 where mask is True
    stack = np.zeros((num, height, width), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        m = np.asarray(mask).squeeze()
        stack[idx] = (m > 0).astype(np.uint8) * 255

    # Ensure .ome.tiff extension
    base, ext = os.path.splitext(output_path)
    if not (output_path.endswith(".ome.tiff") or output_path.endswith(".ome.tif")):
        output_path = f"{base}.ome.tiff"

    metadata = {
        "axes": "CYX",
        "Channel": {"Name": [f"mask_{i}" for i in range(num)]},
    }
    tifffile.imwrite(
        output_path,
        stack,
        ome=True,
        photometric="minisblack",
        compression="deflate",
        metadata=metadata,
    )
    print(
        f"[save_masks_tiff] Wrote {output_path} — {num} channels × {height} × {width}"
    )
    return output_path


# ── optional: quick visualisation ──────────────────────────────────────────
def visualise_masks(image: Image.Image, masks: list[np.ndarray]) -> None:
    """Display *masks* overlaid on *image* using matplotlib."""
    import matplotlib.pyplot as plt

    if not masks:
        print("No masks to show.")
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for idx, mask in enumerate(masks):
        mask_np = np.asarray(mask)
        color_rgb = MASK_COLORS[idx % len(MASK_COLORS)]
        rgba = np.array(color_rgb + [0.6])
        h, w = mask_np.shape[-2:]
        overlay = mask_np.reshape(h, w, 1) * rgba.reshape(1, 1, -1)
        ax.imshow(overlay)

        ys, xs = np.where(mask_np.reshape(h, w))
        if len(ys):
            ax.text(
                int(xs.mean()),
                int(ys.mean()),
                str(idx),
                fontsize=12,
                color="white",
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )

    plt.axis("off")
    plt.title("SAM segmentation on cropped wing")
    plt.show()


# ── convenience: run the full pipeline ─────────────────────────────────────
def run_pipeline(
    image_path: str,
    output_cropped_path: str | None = None,
    output_masks_path: str | None = None,
    text_prompt: str = "wing",
    threshold: float = 0.5,
    overlap_threshold: float = 0.7,
    visualize: bool = True,
    device: str | None = None,
) -> dict | None:
    """Run the full crop → segment → save pipeline.

    Parameters
    ----------
    image_path : str
        Path to the raw input image.
    output_cropped_path : str | None
        If given, save the cropped wing image here.
    output_masks_path : str | None
        If given, save the mask stack as an OME-TIFF here.
    text_prompt : str
        SAM3 text prompt (default ``"wing"``).
    threshold : float
        SAM3 detection threshold.
    overlap_threshold : float
        Minimum wing-overlap ratio for keeping a SAM mask.
    visualize : bool
        Whether to call :func:`visualise_masks` at the end.
    device : str | None
        PyTorch device string.  ``None`` → auto-detect.

    Returns
    -------
    dict | None
        ``{'cropped_image', 'masks', 'wing_mask',
        'cropped_path', 'masks_path'}`` on success, else ``None``.
    """
    device = _resolve_device(device)

    # 1 — crop
    cropped_image, wing_mask = crop_wing(
        image_path,
        text_prompt=text_prompt,
        threshold=threshold,
        device=device,
    )
    if cropped_image is None:
        print("[pipeline] Wing crop failed – aborting.")
        return None

    if output_cropped_path:
        cropped_image.save(output_cropped_path)
        print(f"[pipeline] Saved cropped image → {output_cropped_path}")

    # 2 — segment
    masks = segment_wing(
        cropped_image,
        wing_mask,
        overlap_threshold=overlap_threshold,
        device=device,
    )

    # 3 — save
    masks_path = None
    if output_masks_path and masks:
        masks_path = save_masks_tiff(masks, output_masks_path)

    # 4 — visualise
    if visualize and masks:
        visualise_masks(cropped_image, masks)

    return {
        "cropped_image": cropped_image,
        "masks": masks,
        "wing_mask": wing_mask,
        "cropped_path": output_cropped_path,
        "masks_path": masks_path,
    }


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wing crop + segmentation pipeline")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--cropped", help="Save cropped wing to this path")
    parser.add_argument("--masks", help="Save mask OME-TIFF to this path")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 …")
    parser.add_argument("--prompt", default="wing", help="SAM3 text prompt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--no-viz", action="store_true", help="Disable visualisation")
    args = parser.parse_args()

    run_pipeline(
        args.image,
        output_cropped_path=args.cropped,
        output_masks_path=args.masks,
        text_prompt=args.prompt,
        threshold=args.threshold,
        overlap_threshold=args.overlap,
        visualize=not args.no_viz,
        device=args.device,
    )
