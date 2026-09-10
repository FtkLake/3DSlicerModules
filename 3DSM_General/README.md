# 3DSM_General — 3D Slicer 汎用 Segmentation モジュール (テンプレート)

CT ボリュームを入力にして Segmentation ノードを作る 3D Slicer のスクリプトモジュールの雛形。
「処理 (Segmenter)」を差し替え可能なインターフェースにしてあり、動作確認用の閾値処理 2 種と、
nnU-Net v2 推論 (学習結果フォルダを置くだけで一覧に加わる) が入っている。

このリポジトリにはモデルの重みは含まれていない (`*.pth` は `.gitignore` で除外)。
nnU-Net モデルを使うときは後述の手順でフォルダを置く。

動作確認済み環境: 3D Slicer 5.8.1 (Python 3.9), Windows 11。

## フォルダ構成

```
3DSM_General/
├── GeneralSegmentation/                 <- Slicer に登録するモジュールフォルダ
│   ├── GeneralSegmentation.py           Widget (UI) / Logic (ノード変換) / Test (自己テスト)
│   └── GeneralSegmentationLib/
│       ├── Segmenters.py                Segmentation 処理のインターフェースと実装 (閾値 2 種 + nnU-Net)
│       ├── nnunet_predict_worker.py     nnU-Net 推論ワーカー (PythonSlicer.exe で別プロセス実行)
│       └── nnUNetTemplate/nnUNet_results/   モデルを置くときの雛形 (README.txt 参照。このままでは一覧に出ない)
├── tools/
│   ├── run_slicer.cmd                   モジュールを読み込んだ Slicer を起動 (設定は変更しない)
│   ├── run_tests.cmd                    自己テストを画面なしで実行
│   ├── run_tests.py                     (run_tests.cmd から呼ばれる)
│   ├── install_nnunet.cmd               Slicer の Python に nnunetv2 / torch を入れる (requirements.txt を使う)
│   ├── check_nnunet.py                  同梱モデルで 1 症例を推論し Dice を表示 (下の cmd から呼ばれる)
│   ├── check_nnunet_luna16.cmd          例: LUNA16 モデルを置いたときの確認スクリプト
│   └── check_nnunet_airrc.cmd           例: AirRC モデルを置いたときの確認スクリプト
├── requirements.txt                     Slicer の Python に入れる nnU-Net 推論用パッケージ (通常の Python 用ではない)
├── .gitignore                           __pycache__ / *.pth (モデル重み) などを除外
└── README.md
```

## インストール (別の PC に入れるとき)

### 1. 3D Slicer 5.8.1

https://download.slicer.org/ から Windows 版 5.8.1 を入れる。既定では `%LOCALAPPDATA%\slicer.org\Slicer5.8.1` に入る。
別の場所やバージョンに入れた場合は、`tools\*.cmd` を使う前に環境変数 `SLICER_HOME` にそのフォルダを設定する
(cmd 内の既定値を書き換えてもよい)。

```
set SLICER_HOME=D:\Apps\Slicer 5.8.1
```

### 2. このリポジトリ

任意の場所に clone またはコピーする (パスに日本語や空白が無い場所を推奨)。
`tools\run_slicer.cmd` で起動すればモジュールが読み込まれる。
常に読み込ませたい場合は Slicer の `Edit > Application Settings > Modules > Additional module paths` に
`<リポジトリ>\GeneralSegmentation` を追加して再起動する。

閾値処理 2 種はこれだけで動く。ここで `tools\run_tests.cmd` を実行し、`RESULT: OK` が出ることを確認する。

### 3. nnU-Net (nnU-Net 推論を使う場合のみ)

Slicer の Python (`bin\PythonSlicer.exe`) に nnunetv2 と PyTorch を入れる。方法は 2 つ。どちらか一方でよい。

**A. SlicerNNUNet 拡張を使う (推奨)**
`View > Extension Manager` で `SlicerNNUNet` を入れて再起動し、`Segmentation > nnUNet` モジュールの `nnUNet Install` を押す。
PyTorch 拡張が GPU を検出して合う torch を選び、nnunetv2 も入れてくれる。

**B. requirements.txt で入れる**
`tools\install_nnunet.cmd` を実行する。中で次を実行し、最後に torch / nnunetv2 / numpy のバージョンと CUDA の可否を表示する。

```
<Slicer>\bin\PythonSlicer.exe -m pip install -r requirements.txt
```

`requirements.txt` は Slicer の Python 専用で、通常の Python 環境に入れるものではない。内容は
torch 2.8.0 (CUDA 12.6 版)、nnunetv2 2.5.2、`numpy<2` (Slicer 5.8 の VTK / SimpleITK が numpy 1.x 向けのため)。
GPU が無い PC や NVIDIA ドライバが CUDA 12 に対応していない PC では、requirements.txt の torch の行を
`torch==2.8.0` (PyPI の CPU 版) に変え、`--extra-index-url` の行を消してから実行する。

**確認**
Slicer を起動して General Segmentation モジュールを開き、`nnU-Net v2 (custom model path)` を選んで
Apply ボタンの下に赤字の理由が出ていなければ導入できている。
GPU を使うには NVIDIA ドライバが CUDA 12.x に対応している (525 以降) こと。対応していなくても CPU に自動で落ちる (10 分以上かかる)。

### 4. モデルの配置

nnU-Net の学習結果フォルダを `GeneralSegmentationLib/<モデル名>/nnUNet_results/` に置く (後述「モデルの置き方」)。
モデルの重み (`*.pth`) はリポジトリに含めていないので、別途コピーする。

### 困ったとき

| 症状 | 対処 |
|---|---|
| cmd を実行すると `Slicer.exe` / `PythonSlicer.exe` が見つからない | `SLICER_HOME` を Slicer のインストールフォルダに設定する |
| Apply の下に `nnunetv2 is not installed ...` | 手順 3 を行う。入れたはずなら `PythonSlicer.exe -c "import nnunetv2"` で確認 |
| ログに `CUDA is not available; falling back to CPU.` | torch が CPU 版、または NVIDIA ドライバが古い。`PythonSlicer.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"` で確認 |
| pip 後に Slicer が起動しない / numpy のエラー | numpy が 2.x に上がった可能性。`PythonSlicer.exe -m pip install "numpy<2"` で戻す |
| Segmentation method に `nnU-Net: <モデル名>` が出ない | `plans.json`, `dataset.json`, `fold_*/checkpoint_*.pth` が揃っているか確認 (雛形フォルダは出ない) |

## 使い方

### 起動

`tools\run_slicer.cmd` をダブルクリックする (Slicer 5.8.1 が `%LOCALAPPDATA%\slicer.org\Slicer5.8.1` にある前提。
別の場所なら cmd 内の `SLICER=` を書き換える)。
`--additional-module-paths` でモジュールを読み込むので Slicer の設定は変わらない。

常に読み込ませたい場合は Slicer の `Edit > Application Settings > Modules > Additional module paths` に
このリポジトリの `GeneralSegmentation` フォルダを追加して再起動する。

### 操作

1. CT (DICOM / NIfTI / NRRD) を読み込む。HU 値の CT であること。
2. モジュール `Segmentation > General Segmentation` を開く。
3. 設定する。

   | 項目 | 内容 |
   |---|---|
   | Input CT volume | 入力ボリューム |
   | Output segmentation | 未選択なら `<入力名>_Segmentation` を新規作成。既存を選ぶと中身を置き換える |
   | Segmentation method | 処理の選択。選ぶとその処理のパラメータ欄が下に出る |

4. **Apply** を押す。ログ欄に経過が出て、Segmentation ノードが作られ 3D 表示される。
   結果は Segment Editor で編集でき、Segmentations モジュールから STL / NRRD に書き出せる。

### 用意している処理

| 名前 | 内容 | パラメータ |
|---|---|---|
| Threshold (single label) | HU 範囲 [lower, upper] を 1 セグメントにする | lower (既定 200), upper (3000), segment name |
| CT tissue (multi label) | 肺野 / 軟部組織 / 骨 の 3 セグメント (多ラベル出力の例) | lung upper (-500), soft lower (-100), bone lower (200) |
| nnU-Net: <モデル名> | 同梱したモデルごとに 1 つ並ぶ。セグメント名は dataset.json の labels | folds, checkpoint, device, step size, disable TTA (model path は固定済み) |
| nnU-Net v2 (custom model path) | 任意の nnU-Net 学習結果フォルダで推論 | model path, folds, checkpoint, device, step size, disable TTA |

## 動作確認 (自己テスト)

```
tools\run_tests.cmd
```

画面を出さずに Slicer を起動し、合成 CT (空気中に軟部組織と骨の球) で次を確認して終了する。モデルもダウンロードも不要。

- Threshold: 1 セグメント、名前とボクセル数が numpy の閾値結果と一致
- CT tissue: 3 セグメント (lung / soft_tissue / bone)、それぞれのボクセル数が一致
- 既存の Segmentation ノードを出力に指定すると中身が置き換わる
- isAvailable() が False の処理は RuntimeError になる
- 同梱 nnU-Net モデルの検出、fold・checkpoint の自動選択、dataset.json のラベル読み取り (推論は走らせない)
- checkpoint の無いモデルフォルダ (雛形) は一覧に入らない

Slicer の GUI 上では、モジュール上部の `Reload & Test` ボタン (Developer mode を有効にすると出る) で同じテストが走る。
Python コンソールに `GeneralSegmentation tests passed.` と出れば成功。

## 処理 (Segmenter) の追加方法

`GeneralSegmentationLib/Segmenters.py` で `SegmenterBase` を継承し、`getSegmenters()` のリストに追加する。

```python
class MySegmenter(SegmenterBase):
    name = "My method"                       # コンボボックスに出る名前
    description = "説明"
    parameters = [                           # UI が自動生成される。型: float / int / bool / str / dir / choice
        ParamSpec("threshold", "Threshold (HU)", "float", 100.0, -1000.0, 3000.0),
    ]

    def segment(self, volumeNode, params, progress):
        image = slicer.util.arrayFromVolume(volumeNode)          # (K, J, I) の numpy 配列
        labelmap = (image > params["threshold"]).astype(np.uint8)  # 同じ形、0 = 背景, 1.. = ラベル
        progress("done")                                          # ログ欄に出る
        return SegmentationResult(labelmap=labelmap, labels={1: "my_label"})
```

Logic 側 (`GeneralSegmentationLogic.run`) が、返ってきたラベル配列を一時 LabelMapVolume と色テーブルに変換して
`ImportLabelmapToSegmentationNode` で Segmentation ノードに取り込み、`labels` の名前でセグメント名を付ける。
Segmenter は Slicer のノード操作を一切しなくてよい。

`isAvailable()` を実装すると、環境が整っていないとき (ライブラリ未インストールなど) に Apply を無効化して理由を表示できる。

## nnU-Net v2 推論

### モデルの置き方

nnU-Net の学習結果フォルダを、次の形で `GeneralSegmentationLib` の直下にコピーする。

```
GeneralSegmentationLib/
└── <モデル名>/                              例: LUNA16, AirRC (この名前が一覧に出る)
    └── nnUNet_results/
        └── <DatasetXXX_名前>/
            └── <trainer>__<plans>__<config>/   例: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres
                ├── dataset.json                labels がそのままセグメント名になる
                ├── plans.json
                ├── fold_0/checkpoint_final.pth (または best / latest)
                └── fold_1/...                  複数あれば既定でアンサンブル
```

起動時に自動で見つかり「Segmentation method」に `nnU-Net: <モデル名>` として並ぶ。
`plans.json`, `dataset.json`, `fold_*/checkpoint_*.pth` が揃ったフォルダだけが検出される
(checkpoint の無い `nnUNetTemplate` は雛形なので一覧に出ない)。
`dataset_fingerprint.json` や `training_log_*.txt`, `validation/` は不要で、`.gitignore` でも除外している。

別のモデルを一時的に試すときは `nnU-Net v2 (custom model path)` を選び、Model path に学習結果フォルダを指定する。

### パラメータ

| パラメータ | 既定値 | 内容 |
|---|---|---|
| Model path | 同梱モデルは固定 (custom のみ空欄) | `nnUNet_results`、`Dataset*`、`<trainer>__<plans>__<config>` のどれを指定してもよい (自動で探す) |
| Folds | 空欄 | 空欄 = checkpoint がある fold を全部使ってアンサンブル。`6` のように指定も可 |
| Checkpoint | auto | final > best > latest の順で最初に見つかったもの |
| Device | cuda | GPU が無ければ自動で cpu に落ちる |
| Step size | 0.5 | スライディングウィンドウの重なり |
| Disable TTA | ON | OFF にすると 8 倍遅いが少し精度が上がる |

### 仕組み

1. 入力ボリュームを `<Slicer temp>/GeneralSegmentation_nnunet_*/input/volume_0000.nii.gz` に書き出す
2. Slicer 付属の `PythonSlicer.exe` で `GeneralSegmentationLib/nnunet_predict_worker.py` を別プロセス実行する。
   ワーカーは nnunetv2 の `nnUNetPredictor.predict_from_files_sequential` を呼ぶ (マルチプロセスを使わないので安全)。
   標準出力はそのまま Slicer のログ欄に流れる
3. 出力 `output/volume.nii.gz` を読み込み、`dataset.json` の `labels` でセグメント名を付ける。
   成功したら一時フォルダは削除、失敗したら残してログにパスを出す

**カスタムトレーナーの扱い**: 学習に独自トレーナー (例: `nnUNetTrainer_LUNA16`) を使った checkpoint は、
Slicer 側の nnunetv2 にそのクラスが無い。ワーカーは checkpoint の `trainer_name` が見つからないとき標準 `nnUNetTrainer` の
別名として登録してから読み込むので、学習結果の `nnUNet_results` をそのまま使える
(推論に使うのはネットワーク構造だけなので結果は同じ)。

### 前提

Slicer の Python に nnunetv2 と PyTorch が入っていること (上の「インストール」手順 3)。
入っていないときは Apply が無効になり、理由が表示される。

### 動作確認スクリプト (モデルを置いたあと)

```
tools\check_nnunet_luna16.cmd
tools\check_nnunet_airrc.cmd
```

画面なしで 1 症例を推論し、正解ラベルがあればラベルごとの Dice を出す。
どちらも LUNA16 / AirRC モデルを置いたときの例で、症例のパスは cmd 内の `D:\AIProj\...` を既定にしている。
別の症例やモデルは環境変数で指定する。

| 環境変数 | 内容 |
|---|---|
| `GS_MODEL` | 同梱モデル名 (`LUNA16` など) またはモデルフォルダのパス (必須) |
| `GS_CT` | 入力 CT (.nii.gz など) (必須) |
| `GS_GT` | 正解ラベル (省略可。あれば Dice を出す) |
| `GS_FOLDS` | folds (既定: 空欄 = 自動) |
| `GS_SAVE` | 結果の .seg.nrrd 保存先 (省略可) |

参考値 (RTX 4070 SUPER, 2026-09-10): LUNA16 (fold 0 + 6, 1 症例) 44 s で nodule Dice 0.887、
AirRC (fold 0, 300x300x286) 16 s で airway_lumen 0.976 / airway_wall 0.918 / pulmonary_arteries 0.966 / pulmonary_veins 0.971。

## 設計上の割り切り

- パラメータは Widget 内にだけ保持し、parameter node (シーン保存) には入れていない。
- 処理は同期実行 (Apply 中は UI が止まる)。nnU-Net 推論中もログ欄だけは更新される。途中停止ボタンは無い。
- 既存の Segmentation ノードを出力に選ぶと、処理前に中身を全て消す。
- .ui ファイルを使わず Python でウィジェットを組んでいる (ファイルを減らすため)。
