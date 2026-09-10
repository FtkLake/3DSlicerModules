nnU-Net v2 モデルを同梱するときの雛形フォルダ (このままでは一覧に出ない)

GeneralSegmentationLib/<モデル名>/nnUNet_results/<DatasetXXX_名前>/<trainer>__<plans>__<config>/
の形で、nnU-Net の学習結果フォルダからそのままコピーして置く。

必要なファイル:
  dataset.json               labels がセグメント名になる (この雛形のものは差し替える)
  plans.json                 nnU-Net が生成したもの (ネットワーク構造を含む。手で書かない)
  fold_<n>/checkpoint_final.pth   (または checkpoint_best.pth / checkpoint_latest.pth)

不要なファイル (あっても無視される):
  dataset_fingerprint.json, fold_*/training_log_*.txt, fold_*/validation/, fold_*/progress.png

plans.json と checkpoint が揃ったフォルダだけが起動時に検出され、
Segmentation method の一覧に「nnU-Net: <モデル名>」として並ぶ。
この雛形は plans.json も checkpoint も無いので検出されない。
