# LinkODT-dynamic

用于 time-lapse TIFF 的点状细胞器动态分析的 Windows 桌面程序。它提供掩膜生成与人工修订、实例分割、跨帧追踪、ROI 动态分析和 MP4 导出。此仓库只包含独立的 Module 5，不包含 LinkNex 的其他模块。

## 给使用者

### 下载现成的 Windows 程序

如果仓库的 [Releases](../../releases) 页面提供 `LinkNex-Module5-Windows.zip`，下载并完整解压，双击其中的 `LinkNex-Module5.exe`。**不要只单独复制 EXE**：它依赖同一文件夹里的 `_internal`。第一次运行可能需要等待数秒。分析结果保存在所选数据文件夹的 `processed_output` 中。请先用自己的少量测试数据检查结果。

### 从源码运行

1. 在 Windows 上安装 **64 位 Python 3.10**，安装时勾选将 Python 加入 PATH，或安装 Python Launcher。
2. 点击仓库页面的 **Code → Download ZIP**，完整解压；也可以用 `git clone`。
3. 双击 `INSTALL_WINDOWS.bat`，等待依赖安装完成。
4. 双击 `RUN_Module5.bat` 启动。

命令行方式：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

`open3d` 是可选依赖。需要时另运行 `.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt`；未安装时分析后端使用 NumPy 实现。MP4 导出需要 `imageio-ffmpeg`，已包含在主依赖中。

## 使用流程

加载一个或多个 time-lapse TIFF 或文件夹 → Generate Mask → 在 Mask Review 中检查和修改 → Run Dynamic Analysis → 浏览结果并导出。详细行为见 [模块说明](MODULE5_PUNCTATE_DYNAMICS.md)。

## 已知范围

- 目前面向 Windows；源包曾在 Python 3.10、NumPy 1.26.4、tifffile 2022.4.8 下验证。
- 本仓库不提供示例显微图像或预训练模型。
- 仓库未附授权许可证。未经权利人授权，请勿将代码或素材用于再分发、修改或商业用途。

## 项目结构

- `main.py`：桌面程序入口
- `punctate_dynamics_module.py`：界面及工作流程
- `backends/`：掩膜、实例与动态分析
- `assets/`：程序图标和图形素材
