"""
nnU-Net v2 推論ワーカー。Slicer 本体とは別プロセス (PythonSlicer.exe) で実行する。

NNUNetSegmenter から次のように呼ばれる:
  PythonSlicer.exe nnunet_predict_worker.py --model-folder <trainer__plans__config フォルダ>
        --input-folder <tmp/input> --output-folder <tmp/output> --folds 6 --checkpoint checkpoint_final.pth
        --device cuda --step-size 0.5 [--disable-tta]

学習結果フォルダ (nnUNet_results/Dataset*/<trainer>__<plans>__<config>) をそのまま使える。
カスタムトレーナー (nnUNetTrainer_LUNA16 など) で学習した checkpoint は、Slicer 側の nnunetv2 に
そのクラスが無いので、標準の nnUNetTrainer の別名として登録してから読み込む
(推論に使うのはネットワーク構造だけなので、標準トレーナーで問題ない)。

進捗は標準出力に 1 行ずつ出す (Slicer 側がログ欄に流す)。
"""
import argparse
import os
import sys
import time


def log(msg):
    print(msg, flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-folder", required=True)
    ap.add_argument("--input-folder", required=True)
    ap.add_argument("--output-folder", required=True)
    ap.add_argument("--folds", required=True, help="例: 6 または 0,1,2,3,4")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--step-size", type=float, default=0.5)
    ap.add_argument("--disable-tta", action="store_true")
    return ap.parse_args()


def register_missing_trainer(model_folder, folds, checkpoint_name):
    """checkpoint の trainer_name が nnunetv2 に無ければ nnUNetTrainer の別名として登録する。"""
    import torch
    import nnunetv2
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    ck_path = os.path.join(model_folder, f"fold_{folds[0]}", checkpoint_name)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    trainer_name = ck["trainer_name"]
    search_dir = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    if recursive_find_python_class(search_dir, trainer_name, "nnunetv2.training.nnUNetTrainer") is None:
        setattr(trainer_module, trainer_name, trainer_module.nnUNetTrainer)
        log(f"Trainer '{trainer_name}' not found in nnunetv2; using standard nnUNetTrainer for inference.")
    log(f"Checkpoint: {ck_path} (trainer={trainer_name}, epoch={ck.get('current_epoch')}, "
        f"best EMA pseudo Dice={ck.get('_best_ema')})")


def main():
    args = parse_args()
    t0 = time.time()

    # 環境変数が無いときの警告を抑える (推論では実際には使わない)
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(key, args.model_folder)

    import torch
    log(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log("CUDA is not available; falling back to CPU.")
        device = "cpu"

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    folds = [int(f) if f.strip().isdigit() else f.strip() for f in args.folds.split(",") if f.strip()]
    register_missing_trainer(args.model_folder, folds, args.checkpoint)

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=(device == "cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    log(f"Loading model: {args.model_folder} folds={folds} checkpoint={args.checkpoint} device={device}")
    predictor.initialize_from_trained_model_folder(args.model_folder, use_folds=folds, checkpoint_name=args.checkpoint)
    log(f"Model loaded in {time.time() - t0:.1f} s. Running inference (this may take a while) ...")

    os.makedirs(args.output_folder, exist_ok=True)
    # sequential 版はマルチプロセスを使わないので、サブプロセスから安全に呼べる
    predictor.predict_from_files_sequential(
        args.input_folder, args.output_folder,
        save_probabilities=False, overwrite=True, folder_with_segs_from_prev_stage=None,
    )
    outputs = sorted(os.listdir(args.output_folder))
    log(f"Inference finished in {time.time() - t0:.1f} s. Output: {outputs}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
