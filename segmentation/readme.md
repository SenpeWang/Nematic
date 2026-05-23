# SGNet3 Segmentation

MMSegmentation integration for SGNet3 with nematic field guidance.

## Structure

```
segmentation/
├── model.py                  # MM_SGNetBackbone + SGNetSegmentor registration
├── engine/
│   └── hooks/
│       └── sgnet_train_hook.py   # Custom hook: early stop, best model save, curves
├── models/
│   └── losses/
│       └── nematic_loss.py       # Nematic loss adapter
├── datasets/
│   └── transforms/
│       └── convert_label.py      # ConvertLabel transform
├── utils/
│   ├── visualization.py          # Prediction / nematic field plots
│   ├── four_panel_visualizer.py  # Four-panel test visualization
│   └── plot_utils.py             # Training curve plotting
├── configs/
│   ├── _base_/
│   │   ├── datasets/
│   │   ├── models/
│   │   ├── schedules/
│   │   └── default_runtime.py
│   └── sgnet/
│       ├── upernet_sgnet_neurite.py
│       ├── upernet_sgnet_neuro.py
│       └── upernet_sgnet_sy5y.py
└── tools/
    ├── train.py
    ├── test.py
    ├── dist_train.sh
    ├── dist_test.sh
    ├── slurm_train.sh
    ├── slurm_test.sh
    ├── analysis_tools/
    ├── misc/
    └── model_converters/
```

## Usage

### Train

```bash
python segmentation/tools/train.py segmentation/configs/sgnet/upernet_sgnet_neurite.py
```

### Test

```bash
python segmentation/tools/test.py segmentation/configs/sgnet/upernet_sgnet_neurite.py \
    ./Outputs/train_results/best_dice_model.pth --show-dir ./vis
```

### Distributed Train

```bash
bash segmentation/tools/dist_train.sh segmentation/configs/sgnet/upernet_sgnet_neurite.py 2
```
