# 发布维护说明

这是 Windows PySide6 桌面程序。GitHub Pages 用于静态网页，不适合运行本项目。

## 分享给别人

分享仓库首页可让其他人查看和下载源码。建议分享 Releases 中的 Windows ZIP：使用者完整解压后双击 LinkNex-Module5.exe，无需安装 Python。EXE 必须和 _internal 文件夹一起保留。

## 更新 Windows 下载包

1. 在 Python 3.10 的干净环境安装 requirements.txt 和 PyInstaller。
2. 从仓库根目录构建（命令如下），测试 dist/LinkNex-Module5/LinkNex-Module5.exe。
3. 将整个 dist/LinkNex-Module5 文件夹压缩为 LinkNex-Module5-Windows.zip。
4. 在 GitHub 仓库的 Releases 页面创建新版本标签，上传 ZIP，并发布。

```powershell
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --clean --onedir --windowed --name LinkNex-Module5 --icon assets/linknex_app.ico --add-data "assets;assets" --collect-all imageio_ffmpeg --exclude-module open3d --exclude-module torch main.py
```

如果本机 Python 来自 Conda，可能需要将该环境 Library/bin/ffi.dll 复制进 dist/LinkNex-Module5/_internal 后再测试。官方 Python 3.10 安装通常不需要此步骤。
