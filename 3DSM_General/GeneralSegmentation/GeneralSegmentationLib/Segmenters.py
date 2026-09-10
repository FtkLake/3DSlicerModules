"""
Segmentation 処理のインターフェース定義と実装。

GeneralSegmentation モジュール本体 (Widget / Logic) は Slicer のノード操作だけを担当し、
「CT ボリュームからラベル配列を作る」処理は全てこのファイルの SegmenterBase 派生クラスが担当する。
新しい処理を追加するときは、SegmenterBase を継承して segment() を実装し、
getSegmenters() のリストに登録するだけでよい。

nnU-Net のモデルは、このフォルダ (GeneralSegmentationLib) の直下に
    <モデル名>/nnUNet_results/Dataset*/<trainer>__<plans>__<config>/
の形で置くと自動で見つかり、「nnU-Net: <モデル名>」として一覧に並ぶ (LUNA16, AirRC など)。

入出力の約束:
  入力  : slicer の vtkMRMLScalarVolumeNode (CT, HU 値) と、ParamSpec で宣言したパラメータの dict
  出力  : SegmentationResult
            labelmap : numpy 配列 (K, J, I) = slicer.util.arrayFromVolume() と同じ並び、同じ形。
                       0 = 背景、1.. = ラベル値。dtype は整数型 (uint8 など)。
            labels   : {ラベル値: セグメント名}
            colors   : {ラベル値: (r, g, b)} 0..1。省略可 (無ければ自動配色)
"""
import glob
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ProgressCallback = Callable[[str], None]
LIB_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ParamSpec:
    """Segmenter がユーザーから受け取るパラメータ 1 個の宣言。Widget はこれを見て入力欄を自動生成する。"""
    key: str
    label: str
    type: str  # "float" | "int" | "bool" | "str" | "dir" | "choice"
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[List[str]] = None
    tooltip: str = ""


@dataclass
class SegmentationResult:
    labelmap: np.ndarray
    labels: Dict[int, str]
    colors: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)


class SegmenterBase:
    """全 Segmenter の基底クラス (インターフェース)。"""

    #: コンボボックスに表示する名前 (一意であること)
    name: str = "Base"
    #: UI に表示する簡単な説明
    description: str = ""
    #: ユーザーに入力させるパラメータの一覧
    parameters: List[ParamSpec] = []

    def defaultParameters(self) -> Dict[str, Any]:
        return {p.key: p.default for p in self.parameters}

    def isAvailable(self) -> Tuple[bool, str]:
        """この処理が今の環境で実行できるか。(True, "") または (False, 理由)。"""
        return True, ""

    def segment(self, volumeNode, params: Dict[str, Any], progress: ProgressCallback) -> SegmentationResult:
        """
        volumeNode : 入力 CT (vtkMRMLScalarVolumeNode)
        params     : {ParamSpec.key: 値}
        progress   : 進捗メッセージを渡すコールバック (UI のログに出る)
        """
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------
# 簡単な処理 (一連の動作確認用)
# ----------------------------------------------------------------------------------------------
def _volumeArray(volumeNode) -> np.ndarray:
    import slicer
    return slicer.util.arrayFromVolume(volumeNode)


class ThresholdSegmenter(SegmenterBase):
    """HU 値の範囲 [lower, upper] を 1 つのセグメントにする最も単純な処理。"""

    name = "Threshold (single label)"
    description = "指定した HU 範囲のボクセルを 1 つのセグメントにします。"
    parameters = [
        ParamSpec("lower", "Lower threshold (HU)", "float", 200.0, -2000.0, 5000.0, tooltip="この値以上を対象"),
        ParamSpec("upper", "Upper threshold (HU)", "float", 3000.0, -2000.0, 5000.0, tooltip="この値以下を対象"),
        ParamSpec("segmentName", "Segment name", "str", "Threshold"),
    ]

    def segment(self, volumeNode, params, progress):
        lower, upper = float(params["lower"]), float(params["upper"])
        if lower > upper:
            raise ValueError(f"Lower threshold ({lower}) must be <= upper threshold ({upper}).")
        progress(f"Thresholding {lower} <= HU <= {upper} ...")
        image = _volumeArray(volumeNode)
        labelmap = ((image >= lower) & (image <= upper)).astype(np.uint8)
        progress(f"  {int(labelmap.sum())} voxels selected.")
        return SegmentationResult(
            labelmap=labelmap,
            labels={1: str(params.get("segmentName") or "Threshold")},
            colors={1: (0.9, 0.9, 0.2)},
        )


class CTTissueSegmenter(SegmenterBase):
    """HU 値の固定範囲で肺野 / 軟部組織 / 骨に分ける多ラベルの例。複数セグメント出力の動作確認用。"""

    name = "CT tissue (multi label)"
    description = "HU 値で 肺野 / 軟部組織 / 骨 の 3 セグメントを作ります (多ラベル出力の例)。"
    parameters = [
        ParamSpec("lungUpper", "Lung upper (HU)", "float", -500.0, -1000.0, 0.0, tooltip="肺野: -1000 .. この値"),
        ParamSpec("softLower", "Soft tissue lower (HU)", "float", -100.0, -500.0, 100.0),
        ParamSpec("boneLower", "Bone lower (HU)", "float", 200.0, 0.0, 1500.0, tooltip="骨: この値以上"),
    ]

    def segment(self, volumeNode, params, progress):
        lungUpper = float(params["lungUpper"])
        softLower = float(params["softLower"])
        boneLower = float(params["boneLower"])
        if not (lungUpper < softLower < boneLower):
            raise ValueError("Thresholds must satisfy lungUpper < softLower < boneLower.")
        image = _volumeArray(volumeNode)
        labelmap = np.zeros(image.shape, dtype=np.uint8)
        progress("Lung ...")
        labelmap[(image >= -1000) & (image <= lungUpper)] = 1
        progress("Soft tissue ...")
        labelmap[(image >= softLower) & (image < boneLower)] = 2
        progress("Bone ...")
        labelmap[image >= boneLower] = 3
        return SegmentationResult(
            labelmap=labelmap,
            labels={1: "lung", 2: "soft_tissue", 3: "bone"},
            colors={1: (0.6, 0.8, 1.0), 2: (0.9, 0.6, 0.5), 3: (0.95, 0.95, 0.8)},
        )


# ----------------------------------------------------------------------------------------------
# nnU-Net v2
# ----------------------------------------------------------------------------------------------
CHECKPOINT_PRIORITY = ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth")

#: dataset.json のラベル名に対する既定色 (無い名前は自動配色)
LABEL_COLORS = {
    "nodule": (0.9, 0.3, 0.3),
    "airway_lumen": (0.4, 0.9, 0.9),
    "airway_wall": (0.9, 0.85, 0.3),
    "pulmonary_arteries": (0.9, 0.2, 0.2),
    "pulmonary_veins": (0.2, 0.3, 0.9),
    "lung": (0.6, 0.8, 1.0),
}


@dataclass
class BundledModel:
    """GeneralSegmentationLib/<name>/nnUNet_results/<Dataset*>/<trainer__plans__config> として同梱したモデル。"""
    name: str          # フォルダ名 (LUNA16, AirRC など)
    datasetName: str   # Dataset100_LUNA16 など
    modelFolder: str   # <trainer>__<plans>__<config> フォルダの絶対パス
    labels: Dict[int, str]


def _isModelFolder(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "plans.json")) and os.path.isfile(os.path.join(path, "dataset.json"))


def _readDatasetJson(modelFolder: str) -> dict:
    with open(os.path.join(modelFolder, "dataset.json"), encoding="utf-8") as f:
        return json.load(f)


def _labelsFromDatasetJson(datasetJson: dict) -> Dict[int, str]:
    labels = {}
    for name, value in datasetJson.get("labels", {}).items():
        if isinstance(value, (list, tuple)):  # region-based label はそのままでは扱えないので飛ばす
            continue
        value = int(value)
        if value != 0:
            labels[value] = str(name)
    return labels


def _availableFolds(modelFolder: str) -> List[int]:
    """checkpoint_*.pth を持つ fold_<数字> フォルダの番号一覧。"""
    folds = []
    for d in sorted(glob.glob(os.path.join(modelFolder, "fold_*"))):
        suffix = os.path.basename(d)[5:]
        if os.path.isdir(d) and suffix.isdigit() and glob.glob(os.path.join(d, "checkpoint_*.pth")):
            folds.append(int(suffix))
    return folds


def discoverBundledModels(libDir: str = LIB_DIR) -> List[BundledModel]:
    """
    GeneralSegmentationLib 直下の <name>/nnUNet_results/Dataset*/<model folder> を列挙する。
    plans.json / dataset.json があっても checkpoint が 1 つも無いフォルダ (nnUNetTemplate の雛形など) は
    推論できないので一覧に入れない。
    """
    found = []
    for modelFolder in sorted(glob.glob(os.path.join(libDir, "*", "nnUNet_results", "Dataset*", "*__*__*"))):
        if not os.path.isdir(modelFolder) or not _isModelFolder(modelFolder):
            continue
        if not _availableFolds(modelFolder):
            continue
        datasetDir = os.path.dirname(modelFolder)
        name = os.path.basename(os.path.dirname(os.path.dirname(datasetDir)))
        try:
            labels = _labelsFromDatasetJson(_readDatasetJson(modelFolder))
        except Exception:
            labels = {}
        found.append(BundledModel(name, os.path.basename(datasetDir), modelFolder, labels))
    return found


class NNUNetSegmenter(SegmenterBase):
    """
    nnU-Net v2 推論。学習結果フォルダ (nnUNet_results/Dataset*/...) をそのまま使う。

    流れ:
      1. 入力ボリュームを <tmp>/input/volume_0000.nii.gz に書き出す
      2. Slicer 付属の PythonSlicer.exe で nnunet_predict_worker.py を別プロセス実行する
         (Slicer の Python に nnunetv2 / torch が入っている必要がある。SlicerNNUNet 拡張の「nnUNet Install」で入る)
      3. <tmp>/output/volume.nii.gz を読み込んでラベル配列にし、dataset.json の labels で名前を付けて返す

    bundled を渡すと、そのモデル専用の Segmenter になる (名前と Model path の既定値が固定される)。
    """

    name = "nnU-Net v2 (custom model path)"
    description = ("任意の nnU-Net v2 学習結果フォルダ (nnUNet_results / Dataset* / <trainer>__<plans>__<config>) で推論します。"
                   "GPU で 1 症例 1 分前後、CPU では 10 分以上かかります。実行中は UI が止まります。")

    def __init__(self, bundled: Optional[BundledModel] = None):
        self.bundled = bundled
        modelPath = bundled.modelFolder if bundled else ""
        if bundled:
            labelText = ", ".join(bundled.labels.values()) or "?"
            self.name = f"nnU-Net: {bundled.name}"
            self.description = (f"同梱モデル {bundled.datasetName} ({os.path.basename(bundled.modelFolder)}) で推論します。"
                                f" セグメント: {labelText}。GPU で 1 症例 1 分前後、CPU では 10 分以上かかります。実行中は UI が止まります。")
        self.parameters = [
            ParamSpec("modelPath", "Model path", "dir", modelPath,
                      tooltip="nnUNet_results、Dataset*、または <trainer>__<plans>__<config> フォルダのいずれか"),
            ParamSpec("folds", "Folds", "str", "", tooltip="空欄 = 見つかった fold を全部使う。例: 6 または 0,1,2,3,4"),
            ParamSpec("checkpoint", "Checkpoint", "choice", "auto",
                      choices=["auto"] + list(CHECKPOINT_PRIORITY), tooltip="auto = final > best > latest の順で最初に見つかったもの"),
            ParamSpec("device", "Device", "choice", "cuda", choices=["cuda", "cpu"]),
            ParamSpec("stepSize", "Step size", "float", 0.5, 0.1, 1.0, tooltip="スライディングウィンドウの重なり。小さいほど遅く高精度"),
            ParamSpec("disableTta", "Disable TTA (mirroring)", "bool", True, tooltip="OFF にすると 8 倍遅いが少し精度が上がる"),
        ]

    # ---- 環境 ---------------------------------------------------------------------------------
    @staticmethod
    def _slicerPython() -> str:
        import slicer
        exe = "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer"
        return os.path.join(slicer.app.slicerHome, "bin", exe)

    @staticmethod
    def _nnunetInstalled() -> bool:
        import slicer
        base = os.path.join(slicer.app.slicerHome, "lib", "Python")
        patterns = [os.path.join(base, "Lib", "site-packages", "nnunetv2"),
                    os.path.join(base, "lib", "python3*", "site-packages", "nnunetv2")]
        return any(glob.glob(p) for p in patterns)

    def isAvailable(self):
        if not os.path.isfile(self._slicerPython()):
            return False, f"PythonSlicer not found: {self._slicerPython()}"
        if not self._nnunetInstalled():
            return False, "nnunetv2 is not installed in Slicer's Python (use SlicerNNUNet extension's 'nnUNet Install')."
        return True, ""

    # ---- モデルフォルダの解決 ------------------------------------------------------------------
    @classmethod
    def resolveModelFolder(cls, modelPath: str) -> str:
        """nnUNet_results / Dataset* / <trainer>__<plans>__<config> のどれを渡されても model folder を返す。"""
        modelPath = os.path.abspath(os.path.expanduser(modelPath or ""))
        if not os.path.isdir(modelPath):
            raise FileNotFoundError(f"Model path does not exist: {modelPath}")
        if _isModelFolder(modelPath):
            return modelPath
        found = [d for pattern in ("*", os.path.join("*", "*"))
                 for d in sorted(glob.glob(os.path.join(modelPath, pattern)))
                 if os.path.isdir(d) and "__" in os.path.basename(d) and _isModelFolder(d)]
        if not found:
            raise FileNotFoundError(
                f"No nnU-Net model folder (containing plans.json and dataset.json) found under: {modelPath}")
        return found[0]

    availableFolds = staticmethod(_availableFolds)

    @staticmethod
    def resolveCheckpoint(modelFolder: str, folds: List[int], requested: str) -> str:
        foldDir = os.path.join(modelFolder, f"fold_{folds[0]}")
        candidates = CHECKPOINT_PRIORITY if requested in ("", "auto") else (requested,)
        for name in candidates:
            if all(os.path.isfile(os.path.join(modelFolder, f"fold_{f}", name)) for f in folds):
                return name
        raise FileNotFoundError(f"Checkpoint {list(candidates)} not found in {foldDir} (for folds {folds}).")

    readDatasetJson = staticmethod(_readDatasetJson)
    labelsFromDatasetJson = staticmethod(_labelsFromDatasetJson)

    # ---- 実行 ---------------------------------------------------------------------------------
    def segment(self, volumeNode, params, progress):
        import slicer

        modelFolder = self.resolveModelFolder(str(params["modelPath"]))
        allFolds = self.availableFolds(modelFolder)
        if not allFolds:
            raise FileNotFoundError(f"No fold_* folder with checkpoints in {modelFolder}")
        foldsText = str(params.get("folds") or "").strip()
        folds = [int(f) for f in foldsText.replace(" ", "").split(",") if f] if foldsText else allFolds
        missing = [f for f in folds if f not in allFolds]
        if missing:
            raise FileNotFoundError(f"Requested folds {missing} not found. Available: {allFolds}")
        checkpoint = self.resolveCheckpoint(modelFolder, folds, str(params.get("checkpoint") or "auto"))
        datasetJson = _readDatasetJson(modelFolder)
        fileEnding = datasetJson.get("file_ending", ".nii.gz")
        labels = _labelsFromDatasetJson(datasetJson)
        progress(f"Model folder : {modelFolder}")
        progress(f"Folds        : {folds}   checkpoint: {checkpoint}")
        progress(f"Labels       : {labels}")

        tmpDir = tempfile.mkdtemp(prefix="GeneralSegmentation_nnunet_", dir=slicer.app.temporaryPath)
        inDir, outDir = os.path.join(tmpDir, "input"), os.path.join(tmpDir, "output")
        os.makedirs(inDir)
        os.makedirs(outDir)
        try:
            inputPath = os.path.join(inDir, "volume_0000" + fileEnding)
            progress(f"Exporting volume to {inputPath}")
            if not slicer.util.exportNode(volumeNode, inputPath) or not os.path.isfile(inputPath):
                raise RuntimeError(f"Failed to export volume to {inputPath}")

            worker = os.path.join(LIB_DIR, "nnunet_predict_worker.py")
            cmd = [self._slicerPython(), worker,
                   "--model-folder", modelFolder, "--input-folder", inDir, "--output-folder", outDir,
                   "--folds", ",".join(str(f) for f in folds), "--checkpoint", checkpoint,
                   "--device", str(params.get("device") or "cuda"), "--step-size", str(float(params.get("stepSize", 0.5)))]
            if params.get("disableTta", True):
                cmd.append("--disable-tta")
            progress("Running: " + " ".join(cmd))
            self._runWorker(cmd, progress)

            outputPath = os.path.join(outDir, "volume" + fileEnding)
            if not os.path.isfile(outputPath):
                raise RuntimeError(f"nnU-Net produced no output file: {outputPath}")
            progress(f"Loading result {outputPath}")
            labelmap = self._loadLabelmapArray(outputPath)
            expected = slicer.util.arrayFromVolume(volumeNode).shape
            if labelmap.shape != expected:
                raise RuntimeError(f"Result shape {labelmap.shape} != input shape {expected}")
            present = sorted(int(v) for v in np.unique(labelmap) if v != 0)
            for v in present:
                labels.setdefault(v, f"label_{v}")
            progress("Voxels per label: " + ", ".join(f"{labels[v]}={int((labelmap == v).sum())}" for v in present))
        except Exception:
            progress(f"Temporary files kept for debugging: {tmpDir}")
            raise
        shutil.rmtree(tmpDir, ignore_errors=True)
        colors = {v: LABEL_COLORS[n] for v, n in labels.items() if n in LABEL_COLORS}
        return SegmentationResult(labelmap=labelmap, labels=labels, colors=colors)

    @staticmethod
    def _runWorker(cmd: List[str], progress: ProgressCallback):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        tail = []
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    progress("  " + line)
                    tail.append(line)
            code = proc.wait()
        except BaseException:
            proc.kill()
            raise
        if code != 0:
            raise RuntimeError("nnU-Net worker failed (exit code %d):\n%s" % (code, "\n".join(tail[-15:])))

    @staticmethod
    def _loadLabelmapArray(path: str) -> np.ndarray:
        import slicer
        node = slicer.util.loadVolume(path, {"labelmap": True, "show": False})
        try:
            return np.array(slicer.util.arrayFromVolume(node)).astype(np.int16)
        finally:
            slicer.mrmlScene.RemoveNode(node)


# ----------------------------------------------------------------------------------------------
# 登録
# ----------------------------------------------------------------------------------------------
def getSegmenters() -> List[SegmenterBase]:
    """UI に並べる Segmenter の一覧。新しい処理はここに追加する。同梱 nnU-Net モデルは自動で並ぶ。"""
    segmenters: List[SegmenterBase] = [
        ThresholdSegmenter(),
        CTTissueSegmenter(),
    ]
    segmenters += [NNUNetSegmenter(m) for m in discoverBundledModels()]
    segmenters.append(NNUNetSegmenter())
    return segmenters


def findSegmenter(name: str) -> Optional[SegmenterBase]:
    for s in getSegmenters():
        if s.name == name:
            return s
    return None
