# Punctate Organelle Dynamics

Module 5 uses one central **Working View** for the complete time-lapse workflow.

## Main viewer

There is no separate Source Preview. The main frame slider browses:

- the original unsplit time-lapse before Split;
- the stacked split-image time-lapse after Split;
- the stacked mask time-lapse in **Mask Review**;
- the stacked instance time-lapse in **Image + Instance**;
- the same instance stack plus persistent trajectories in **Dynamic Overlay**.

For fast browsing LinkNex creates single TIFF viewer stacks in `processed_output`:

```text
viewer_original_stack.tif
viewer_mask_stack.tif
viewer_instance_rgb_stack.tif
```

The stacks are read as single time-lapse arrays/memory maps so changing the main slider does not repeatedly reopen individual TIFF/TXT files.

## Common background

All three main display modes use the same selected background:

- Original image
- Black
- White
- Gray
- a registered second-channel time-lapse loaded with **Choose Background Image**

`Overlay OFF` shows only this common background.

## Mask display

Mask overlay opacity is fixed at 35%. There is no mask-opacity slider. Manual  /  editing remains available in Mask Review. Saving or auto-saving a corrected mask also updates the corresponding frame in `viewer_mask_stack.tif`.

## Dynamic analysis

**Run Dynamic Analysis** performs reviewed-mask conversion, instance segmentation, persistent tracking, temporal network analysis, and then creates `viewer_instance_rgb_stack.tif` for fast display.

The separate Dynamic Analysis Log reports processing stages, frame progress, saved outputs and paths. A permanent copy is stored at:

```text
processed_output/dynamic_analysis.log
```

## ROI output

ROI results are stored directly in:

```text
processed_output/ROI_results/
├─ roi_dynamic_metrics.csv
├─ roi_organelle_count.png
└─ roi_mean_intensity.png
```

The two PNG time courses use 2.0 line width and 2.0 plot-frame width.

## Brush editing fix (single-stack viewer)

In the single-stack viewer,  /  now automatically switches to Mask Review, forces Overlay ON, reloads the current frame mask, disables ROI mouse interception during painting, and auto-saves each completed stroke to both the frame mask and `viewer_mask_stack.tif`. If no mask exists, LinkNex reports this explicitly instead of silently ignoring the brush action.
