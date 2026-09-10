"""
GeneralSegmentation - CT ボリュームに対する汎用 Segmentation モジュール (3D Slicer scripted module)

構成:
  GeneralSegmentationWidget : 入力ボリューム / 処理 (Segmenter) / パラメータ / 出力を選んで Apply する UI
  GeneralSegmentationLogic  : Segmenter を呼び出し、返ってきたラベル配列を Segmentation ノードに変換する
  GeneralSegmentationTest   : 合成 CT で一連の動作を確認する自己テスト (Reload and Test ボタン)

Segmentation 処理そのものは GeneralSegmentationLib/Segmenters.py に分離してある。
"""
import logging
import os
import time
from typing import Dict, Optional

import ctk
import numpy as np
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from GeneralSegmentationLib import (
    ParamSpec,
    SegmentationResult,
    SegmenterBase,
    getSegmenters,
)


class GeneralSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "General Segmentation"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Thoracentes"]
        self.parent.helpText = (
            "CT ボリュームに対する汎用 Segmentation モジュール。<br>"
            "処理 (Segmenter) を選び、パラメータを設定して Apply すると Segmentation ノードが作られます。<br>"
            "処理は GeneralSegmentationLib/Segmenters.py で追加できます。"
            "nnU-Net v2 のモデルは GeneralSegmentationLib/&lt;モデル名&gt;/nnUNet_results/ に置くと自動で一覧に並びます。"
        )
        self.parent.acknowledgementText = "Structure loosely follows the SlicerNNUNet extension (Kitware SAS)."


# ==============================================================================================
# Widget
# ==============================================================================================
class GeneralSegmentationWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic: Optional[GeneralSegmentationLogic] = None
        self._segmenters = []
        self._paramWidgets: Dict[str, qt.QWidget] = {}
        self._paramSpecs: Dict[str, ParamSpec] = {}

    # ---- UI 構築 -------------------------------------------------------------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = GeneralSegmentationLogic()
        self._segmenters = getSegmenters()

        # 入出力
        ioBox = ctk.ctkCollapsibleButton()
        ioBox.text = "Input / Output"
        self.layout.addWidget(ioBox)
        ioForm = qt.QFormLayout(ioBox)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("入力 CT ボリューム (HU 値)")
        ioForm.addRow("Input CT volume:", self.inputSelector)

        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.renameEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.noneDisplay = "(Create new segmentation)"
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("出力 Segmentation ノード。未選択なら新規作成。既存を選ぶと中身は置き換えられる")
        ioForm.addRow("Output segmentation:", self.outputSelector)

        # 処理の選択とパラメータ
        procBox = ctk.ctkCollapsibleButton()
        procBox.text = "Segmentation method"
        self.layout.addWidget(procBox)
        procLayout = qt.QVBoxLayout(procBox)

        self.segmenterCombo = qt.QComboBox()
        for s in self._segmenters:
            self.segmenterCombo.addItem(s.name)
        procLayout.addWidget(self.segmenterCombo)

        self.descriptionLabel = qt.QLabel()
        self.descriptionLabel.wordWrap = True
        self.descriptionLabel.styleSheet = "color: gray;"
        procLayout.addWidget(self.descriptionLabel)

        self.paramGroup = qt.QGroupBox("Parameters")
        self.paramForm = qt.QFormLayout(self.paramGroup)
        procLayout.addWidget(self.paramGroup)

        # 実行
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "選択した処理を実行して Segmentation を作成"
        self.applyButton.enabled = False
        self.layout.addWidget(self.applyButton)

        self.statusLabel = qt.QLabel("")
        self.layout.addWidget(self.statusLabel)

        logBox = ctk.ctkCollapsibleButton()
        logBox.text = "Log"
        logBox.collapsed = False
        self.layout.addWidget(logBox)
        logLayout = qt.QVBoxLayout(logBox)
        self.logText = qt.QPlainTextEdit()
        self.logText.readOnly = True
        self.logText.maximumBlockCount = 2000
        logLayout.addWidget(self.logText)

        self.layout.addStretch(1)

        # 接続
        self.inputSelector.currentNodeChanged.connect(self._updateApplyState)
        self.segmenterCombo.currentIndexChanged.connect(self._onSegmenterChanged)
        self.applyButton.clicked.connect(self.onApply)

        self._onSegmenterChanged()
        self._updateApplyState()

    def cleanup(self):
        pass

    # ---- 処理の切り替えに応じてパラメータ欄を作り直す ---------------------------------------
    def currentSegmenter(self) -> SegmenterBase:
        return self._segmenters[self.segmenterCombo.currentIndex]

    def _onSegmenterChanged(self, *_):
        segmenter = self.currentSegmenter()
        self.descriptionLabel.text = segmenter.description
        self.statusLabel.text = ""
        self._clearForm(self.paramForm)
        self._paramWidgets = {}
        self._paramSpecs = {}
        for spec in segmenter.parameters:
            w = self._createParamWidget(spec)
            self._paramWidgets[spec.key] = w
            self._paramSpecs[spec.key] = spec
            self.paramForm.addRow(spec.label + ":", w)
        self.paramGroup.visible = bool(segmenter.parameters)
        self._updateApplyState()

    @staticmethod
    def _clearForm(form):
        while form.count():
            item = form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    @staticmethod
    def _createParamWidget(spec: ParamSpec):
        if spec.type == "float":
            w = qt.QDoubleSpinBox()
            w.decimals = 1
            w.setRange(spec.minimum if spec.minimum is not None else -1e9,
                       spec.maximum if spec.maximum is not None else 1e9)
            w.value = float(spec.default)
        elif spec.type == "int":
            w = qt.QSpinBox()
            w.setRange(int(spec.minimum) if spec.minimum is not None else -10 ** 9,
                       int(spec.maximum) if spec.maximum is not None else 10 ** 9)
            w.value = int(spec.default)
        elif spec.type == "bool":
            w = qt.QCheckBox()
            w.checked = bool(spec.default)
        elif spec.type == "dir":
            w = ctk.ctkPathLineEdit()
            w.filters = ctk.ctkPathLineEdit.Dirs
            w.currentPath = str(spec.default)
        elif spec.type == "choice":
            w = qt.QComboBox()
            for c in spec.choices or []:
                w.addItem(c)
            w.currentText = str(spec.default)
        else:  # "str"
            w = qt.QLineEdit()
            w.text = str(spec.default)
        if spec.tooltip:
            w.toolTip = spec.tooltip
        return w

    def currentParameters(self) -> Dict[str, object]:
        values = {}
        for key, w in self._paramWidgets.items():
            t = self._paramSpecs[key].type
            if t in ("float", "int"):
                values[key] = w.value
            elif t == "bool":
                values[key] = w.checked
            elif t == "dir":
                values[key] = w.currentPath
            elif t == "choice":
                values[key] = w.currentText
            else:
                values[key] = w.text
        return values

    def _updateApplyState(self, *_):
        available, reason = self.currentSegmenter().isAvailable()
        hasInput = self.inputSelector.currentNode() is not None
        self.applyButton.enabled = hasInput and available
        if not available:
            self.statusLabel.text = reason
        elif not hasInput:
            self.statusLabel.text = "Select an input CT volume."
        # 実行可能なときはステータス ("Done." など) をそのまま残す

    # ---- 実行 ---------------------------------------------------------------------------------
    def log(self, msg: str):
        self.logText.appendPlainText(msg)
        self.logText.verticalScrollBar().setValue(self.logText.verticalScrollBar().maximum)
        slicer.app.processEvents()

    def onApply(self):
        self.logText.clear()
        inputVolume = self.inputSelector.currentNode()
        outputSeg = self.outputSelector.currentNode()
        segmenter = self.currentSegmenter()
        params = self.currentParameters()
        self.applyButton.enabled = False
        self.statusLabel.text = "Running ..."
        try:
            with slicer.util.tryWithErrorDisplay("Segmentation failed.", waitCursor=True):
                outputSeg = self.logic.run(inputVolume, segmenter, params, outputSeg, progress=self.log)
                self.outputSelector.setCurrentNode(outputSeg)
                self.statusLabel.text = "Done."
        finally:
            self._updateApplyState()


# ==============================================================================================
# Logic
# ==============================================================================================
class GeneralSegmentationLogic(ScriptedLoadableModuleLogic):
    """Segmenter を実行し、結果のラベル配列を Segmentation ノードにする。"""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def run(self, inputVolume, segmenter: SegmenterBase, params: Optional[Dict] = None,
            outputSegmentation=None, progress=None):
        """
        inputVolume        : vtkMRMLScalarVolumeNode
        segmenter          : SegmenterBase 派生インスタンス
        params             : パラメータ dict (None なら既定値)
        outputSegmentation : vtkMRMLSegmentationNode。None なら新規作成。既存なら中身を置き換える
        progress           : str を受け取るコールバック (省略可)
        戻り値             : 結果の vtkMRMLSegmentationNode
        """
        progress = progress or (lambda msg: logging.info(msg))
        if inputVolume is None or inputVolume.GetImageData() is None:
            raise ValueError("Input volume is empty.")
        available, reason = segmenter.isAvailable()
        if not available:
            raise RuntimeError(reason)

        fullParams = segmenter.defaultParameters()
        fullParams.update(params or {})

        progress(f"Method : {segmenter.name}")
        progress(f"Input  : {inputVolume.GetName()}  {inputVolume.GetImageData().GetDimensions()}")
        progress(f"Params : {fullParams}")
        t0 = time.time()
        result = segmenter.segment(inputVolume, fullParams, progress)
        progress(f"Segmenter finished in {time.time() - t0:.1f} s")

        self._validateResult(result, inputVolume)

        if outputSegmentation is None:
            outputSegmentation = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", inputVolume.GetName() + "_Segmentation")
        outputSegmentation.CreateDefaultDisplayNodes()
        outputSegmentation.GetSegmentation().RemoveAllSegments()
        outputSegmentation.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        progress("Importing labelmap into segmentation node ...")
        self._importLabelmap(result, inputVolume, outputSegmentation)

        outputSegmentation.CreateClosedSurfaceRepresentation()
        n = outputSegmentation.GetSegmentation().GetNumberOfSegments()
        progress(f"Done: {n} segment(s) in '{outputSegmentation.GetName()}' ({time.time() - t0:.1f} s total)")
        return outputSegmentation

    @staticmethod
    def _validateResult(result: SegmentationResult, inputVolume):
        expected = slicer.util.arrayFromVolume(inputVolume).shape
        if not isinstance(result, SegmentationResult):
            raise TypeError("Segmenter must return SegmentationResult.")
        if result.labelmap.shape != expected:
            raise ValueError(f"Labelmap shape {result.labelmap.shape} != input shape {expected}")
        if not np.issubdtype(result.labelmap.dtype, np.integer):
            raise TypeError(f"Labelmap dtype must be integer, got {result.labelmap.dtype}")

    @staticmethod
    def _importLabelmap(result: SegmentationResult, inputVolume, segmentationNode):
        """ラベル配列 -> 一時 LabelMapVolume (+色テーブル) -> Segmentation ノードへ取り込み。"""
        labelmap = result.labelmap
        if labelmap.max() > 32767:
            raise ValueError("Too many labels for int16 labelmap.")
        labelmap = labelmap.astype(np.int16, copy=False)

        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "GeneralSegmentationTmpLabel")
        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", "GeneralSegmentationTmpColors")
        try:
            slicer.util.updateVolumeFromArray(labelmapNode, labelmap)
            labelmapNode.CopyOrientation(inputVolume)
            labelmapNode.CreateDefaultDisplayNodes()

            # ラベル値 -> 名前 / 色 を色テーブルにして、取り込み時にセグメント名として使わせる
            maxLabel = int(max([0] + [int(v) for v in result.labels.keys()]))
            colorNode.SetTypeToUser()
            colorNode.SetNumberOfColors(maxLabel + 1)
            colorNode.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
            for value, name in result.labels.items():
                r, g, b = result.colors.get(value) or _defaultColor(value)
                colorNode.SetColor(int(value), str(name), float(r), float(g), float(b), 1.0)
            labelmapNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

            ok = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)
            if not ok:
                raise RuntimeError("ImportLabelmapToSegmentationNode failed.")

            # 念のためラベル値からセグメント名 / 色を確定させる
            segmentation = segmentationNode.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segment = segmentation.GetNthSegment(i)
                value = segment.GetLabelValue()
                if value in result.labels:
                    segment.SetName(str(result.labels[value]))
                    segment.SetColor(*(result.colors.get(value) or _defaultColor(value)))
        finally:
            slicer.mrmlScene.RemoveNode(labelmapNode)
            slicer.mrmlScene.RemoveNode(colorNode)


def _defaultColor(value: int):
    """ラベル値から適当に安定した色を作る (色指定が無いとき用)。"""
    palette = [
        (0.9, 0.3, 0.3), (0.3, 0.8, 0.3), (0.3, 0.5, 0.9), (0.9, 0.8, 0.2),
        (0.8, 0.4, 0.9), (0.3, 0.8, 0.8), (0.9, 0.6, 0.3), (0.6, 0.6, 0.6),
    ]
    return palette[(int(value) - 1) % len(palette)]


# ==============================================================================================
# Test
# ==============================================================================================
class GeneralSegmentationTest(ScriptedLoadableModuleTest):
    """合成 CT (空気の中に球状の '骨' と '軟部組織') で一連の動作を確認する。ダウンロード不要。"""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ThresholdSegmenter()
        self.test_CTTissueSegmenter()
        self.test_ReuseOutputNode()
        self.test_UnavailableSegmenter()
        self.test_NNUNetSegmenterHelpers()
        self.test_BundledModelDiscovery()
        print("GeneralSegmentation tests passed.")

    @staticmethod
    def createSyntheticCT(shape=(40, 64, 64), spacing=(1.0, 1.0, 2.0)):
        """K,J,I の順。背景 -1000 HU、中央に半径 20 の球 (軟部 40 HU)、その中に半径 8 の球 (骨 700 HU)。"""
        k, j, i = np.indices(shape)
        ck, cj, ci = [s // 2 for s in shape]
        r2 = ((k - ck) * spacing[2]) ** 2 + ((j - cj) * spacing[1]) ** 2 + ((i - ci) * spacing[0]) ** 2
        image = np.full(shape, -1000, dtype=np.int16)
        image[r2 <= 20 ** 2] = 40
        image[r2 <= 8 ** 2] = 700
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "SyntheticCT")
        slicer.util.updateVolumeFromArray(volumeNode, image)
        volumeNode.SetSpacing(*spacing)
        volumeNode.SetOrigin(-30, -30, -40)
        volumeNode.CreateDefaultDisplayNodes()
        return volumeNode, image

    @staticmethod
    def segmentVoxelCount(segmentationNode, segmentName, referenceVolume) -> int:
        segId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        arr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segId, referenceVolume)
        return int(np.count_nonzero(arr))

    def test_ThresholdSegmenter(self):
        from GeneralSegmentationLib import ThresholdSegmenter
        volumeNode, image = self.createSyntheticCT()
        logic = GeneralSegmentationLogic()
        seg = logic.run(volumeNode, ThresholdSegmenter(), {"lower": 200, "upper": 3000, "segmentName": "bone"})
        self.assertIsNotNone(seg)
        self.assertEqual(seg.GetSegmentation().GetNumberOfSegments(), 1)
        self.assertEqual(seg.GetSegmentation().GetNthSegment(0).GetName(), "bone")
        expected = int((image >= 200).sum())
        self.assertEqual(self.segmentVoxelCount(seg, "bone", volumeNode), expected)
        print(f"  Threshold: bone voxels = {expected}")

    def test_CTTissueSegmenter(self):
        from GeneralSegmentationLib import CTTissueSegmenter
        volumeNode, image = self.createSyntheticCT()
        logic = GeneralSegmentationLogic()
        seg = logic.run(volumeNode, CTTissueSegmenter())
        names = [seg.GetSegmentation().GetNthSegment(i).GetName()
                 for i in range(seg.GetSegmentation().GetNumberOfSegments())]
        self.assertEqual(names, ["lung", "soft_tissue", "bone"])
        self.assertEqual(self.segmentVoxelCount(seg, "lung", volumeNode), int((image <= -500).sum()))
        self.assertEqual(self.segmentVoxelCount(seg, "soft_tissue", volumeNode),
                         int(((image >= -100) & (image < 200)).sum()))
        self.assertEqual(self.segmentVoxelCount(seg, "bone", volumeNode), int((image >= 200).sum()))
        print(f"  CT tissue: segments = {names}")

    def test_ReuseOutputNode(self):
        from GeneralSegmentationLib import ThresholdSegmenter, CTTissueSegmenter
        volumeNode, _ = self.createSyntheticCT()
        logic = GeneralSegmentationLogic()
        seg = logic.run(volumeNode, CTTissueSegmenter())
        seg2 = logic.run(volumeNode, ThresholdSegmenter(), outputSegmentation=seg)
        self.assertIs(seg, seg2)
        self.assertEqual(seg.GetSegmentation().GetNumberOfSegments(), 1)

    def test_UnavailableSegmenter(self):
        """isAvailable() が False の処理は run() で RuntimeError になる。"""
        from GeneralSegmentationLib import ThresholdSegmenter

        class Unavailable(ThresholdSegmenter):
            def isAvailable(self):
                return False, "not available (test)"

        volumeNode, _ = self.createSyntheticCT()
        logic = GeneralSegmentationLogic()
        with self.assertRaises(RuntimeError):
            logic.run(volumeNode, Unavailable())

    def test_NNUNetSegmenterHelpers(self):
        """nnU-Net 推論そのものは走らせず (時間がかかる)、同梱モデルの検出とフォルダ解決だけ確認する。"""
        from GeneralSegmentationLib import NNUNetSegmenter, discoverBundledModels, getSegmenters
        with self.assertRaises(FileNotFoundError):
            NNUNetSegmenter.resolveModelFolder("D:/this/folder/does/not/exist")
        labels = NNUNetSegmenter.labelsFromDatasetJson({"labels": {"background": 0, "nodule": 1, "region": [1, 2]}})
        self.assertEqual(labels, {1: "nodule"})
        bundled = discoverBundledModels()
        names = [m.name for m in bundled]
        print(f"  bundled nnU-Net models: {names}")
        for m in bundled:
            self.assertEqual(NNUNetSegmenter.resolveModelFolder(m.modelFolder), m.modelFolder)
            folds = NNUNetSegmenter.availableFolds(m.modelFolder)
            self.assertTrue(folds, f"no folds found for {m.name}")
            ck = NNUNetSegmenter.resolveCheckpoint(m.modelFolder, folds, "auto")
            print(f"    {m.name}: {m.datasetName} folds={folds} checkpoint={ck} labels={m.labels}")
        segmenterNames = [s.name for s in getSegmenters()]
        for m in bundled:
            self.assertIn(f"nnU-Net: {m.name}", segmenterNames)
        self.assertIn("nnU-Net v2 (custom model path)", segmenterNames)

    def test_BundledModelDiscovery(self):
        """checkpoint の無いモデルフォルダ (雛形) は一覧に入らず、checkpoint があれば入る。"""
        import json
        import tempfile
        import shutil
        from GeneralSegmentationLib import discoverBundledModels

        libDir = tempfile.mkdtemp(prefix="GeneralSegmentation_discover_")
        try:
            def makeModel(name, withCheckpoint):
                folder = os.path.join(libDir, name, "nnUNet_results", "Dataset100_X",
                                      "nnUNetTrainer__nnUNetPlans__3d_fullres")
                os.makedirs(os.path.join(folder, "fold_0"))
                with open(os.path.join(folder, "plans.json"), "w") as f:
                    json.dump({}, f)
                with open(os.path.join(folder, "dataset.json"), "w") as f:
                    json.dump({"labels": {"background": 0, "target": 1}}, f)
                if withCheckpoint:
                    open(os.path.join(folder, "fold_0", "checkpoint_final.pth"), "wb").close()
                return folder

            makeModel("NoCheckpoint", False)
            real = makeModel("WithCheckpoint", True)
            found = discoverBundledModels(libDir)
            self.assertEqual([m.name for m in found], ["WithCheckpoint"])
            self.assertEqual(found[0].modelFolder, real)
            self.assertEqual(found[0].labels, {1: "target"})
        finally:
            shutil.rmtree(libDir, ignore_errors=True)

