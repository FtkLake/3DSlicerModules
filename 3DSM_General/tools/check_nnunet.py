"""
同梱 nnU-Net モデルで 1 症例を推論し、正解ラベルがあればラベルごとの Dice を出す (画面なし)。
tools/check_nnunet_luna16.cmd / check_nnunet_airrc.cmd から呼ぶ。終了コード 0 = 成功。

環境変数:
  GS_MODEL   同梱モデル名 (LUNA16 / AirRC) またはモデルフォルダのパス   (必須)
  GS_CT      入力 CT (.nii.gz など)                                   (必須)
  GS_GT      正解ラベル (省略可)
  GS_FOLDS   folds (既定: "" = 自動)
  GS_SAVE    結果の .seg.nrrd 保存先 (省略可)
"""
import os
import sys
import time
import traceback

import numpy as np
import slicer


def main():
    import GeneralSegmentation as GS
    from GeneralSegmentationLib import NNUNetSegmenter, discoverBundledModels

    modelSpec = os.environ["GS_MODEL"]
    ctPath = os.environ["GS_CT"]
    gtPath = os.environ.get("GS_GT", "")

    bundled = {m.name: m for m in discoverBundledModels()}
    print("bundled models:", list(bundled))
    if modelSpec in bundled:
        segmenter = NNUNetSegmenter(bundled[modelSpec])
        params = {}
    else:
        segmenter = NNUNetSegmenter()
        params = {"modelPath": modelSpec}
    params["folds"] = os.environ.get("GS_FOLDS", "")
    print("segmenter:", segmenter.name, "| available:", segmenter.isAvailable())

    print("Loading CT:", ctPath)
    ct = slicer.util.loadVolume(ctPath)
    print("CT dimensions:", ct.GetImageData().GetDimensions(), "spacing:", ct.GetSpacing())
    t0 = time.time()
    seg = GS.GeneralSegmentationLogic().run(ct, segmenter, params, progress=lambda m: print(m, flush=True))
    print(f"Total {time.time() - t0:.1f} s; segments = {seg.GetSegmentation().GetNumberOfSegments()}")

    gt = None
    if gtPath and os.path.isfile(gtPath):
        gtNode = slicer.util.loadVolume(gtPath, {"labelmap": True, "show": False})
        gt = slicer.util.arrayFromVolume(gtNode)
        print("GT labels present:", sorted(int(v) for v in np.unique(gt)))

    for i in range(seg.GetSegmentation().GetNumberOfSegments()):
        s = seg.GetSegmentation().GetNthSegment(i)
        value = s.GetLabelValue()
        pred = slicer.util.arrayFromSegmentBinaryLabelmap(seg, seg.GetSegmentation().GetNthSegmentID(i), ct) > 0
        line = f"  segment[{i}] name={s.GetName()} labelValue={value} voxels={int(pred.sum())}"
        if gt is not None:
            g = gt == value
            inter = np.logical_and(g, pred).sum()
            dice = 2 * inter / max(1, g.sum() + pred.sum())
            line += f" | GT voxels={int(g.sum())} Dice={dice:.3f}"
        print(line)

    if os.environ.get("GS_SAVE"):
        slicer.util.saveNode(seg, os.environ["GS_SAVE"])
        print("saved:", os.environ["GS_SAVE"])


try:
    main()
    print("RESULT: OK")
    code = 0
except Exception:
    traceback.print_exc()
    print("RESULT: FAILED")
    code = 1
sys.stdout.flush()
slicer.util.exit(code)
