The pretrained .pkl checkpoint is intentionally not bundled in the ZIP because model weights are large.
From the project root, run:
  python download_model.py --model afhqcat
This saves models/afhqcat.pkl. Use only trusted checkpoints because Python pickle files may execute code during loading.
