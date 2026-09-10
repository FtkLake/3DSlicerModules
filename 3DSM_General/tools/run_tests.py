"""
Slicer 上で GeneralSegmentation の自己テストを実行して終了する (ヘッドレス動作確認用)。

  Slicer.exe --no-splash --no-main-window --testing ^
      --additional-module-paths <repo>/GeneralSegmentation ^
      --python-script <repo>/tools/run_tests.py

終了コード 0 = 成功、1 = 失敗。tools/run_tests.cmd から呼ぶ。
"""
import sys
import traceback

import slicer


def main():
    try:
        import GeneralSegmentation
        print("Module file:", GeneralSegmentation.__file__)
        GeneralSegmentation.GeneralSegmentationTest().runTest()
        print("RESULT: OK")
        code = 0
    except Exception:
        traceback.print_exc()
        print("RESULT: FAILED")
        code = 1
    sys.stdout.flush()
    slicer.util.exit(code)


main()
