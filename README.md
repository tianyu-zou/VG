<p align="center">
  <h1 align="center">Evaluation of Visual Grounding Methods</h1>
  <h3 align="center">For ICCV 2025 MARS2 Workshop</h3>
<p>

## Folder Structure

Each method is organized as a separate folder with the same name as the method.

## Evaluation

To evaluate each method, run the following:

```bash
cd $Method_name
python eval_for_MARS2.py
```

## Notes on Evaluation and Dataset Usage

- Please refer to `TransVG-main/datasets/data_loader_for_MARS2.py` for the correct way to load the MARS2 dataset.
- You can modify your evaluation script based on `TransVG-main/eval_for_MARS2.py`, and save the predicted bounding boxes in **xywh** format.
- If you want to visualize the predicted boxes on the original images, please refer to `TransVG-main/show_origin_box.py`.
- After generating the predictions, use `trans_to_origin_box.py` to convert the **xywh** format boxes into the **x1y1x2y2** format required for MARS2 evaluation.
- Each method is generated a file named `${method_name}_prediction.json` in **xywh** format within its own folder.
- The converted prediction in **x1y1x2y2** format is saved under the `VG/` directory as `${method_name}_prediction.json`.

## Example Workflow

```bash
# Step 1: Run evaluation script to get xywh predictions
cd TransVG
python eval_for_MARS2.py

# Step 2: (Optional) Visualize the predicted boxes
python show_origin_box.py

# Step 3: Convert xywh boxes to x1y1x2y2 format for evaluation
python trans_to_origin_box.py     --input TransVG/TransVG_prediction.json     --output VG/TransVG_prediction.json
```

Make sure to adjust paths and filenames to match your method's folder and output.
