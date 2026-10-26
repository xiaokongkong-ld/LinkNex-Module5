# 发布维护说明

此项目是 PySide6 Windows 桌面程序。GitHub Pages 只能发布静态网页，不能运行它。

## 首次分享

将仓库设为公开后，普通使用者可通过仓库首页的 Code → Download ZIP 获取源码。要提供不需要安装 Python 的版本，请推送版本标签（例如 v0.1.0）。.github/workflows/windows-release.yml 会在 Windows runner 上构建，并把 LinkNex-Module5-Windows.zip 发布到 Releases。

## 后续更新

修改和验证源码后提交。为新版本创建并推送新的 v 开头标签（例如 v0.1.1），等待 Actions 成功完成，再分享对应的 Release 页面链接。

## 本地构建

在已安装依赖及 PyInstaller 的 Python 3.10 环境中，从仓库根目录运行：

```powershell
python -m PyInstaller --noconfirm --clean --onedir --windowed --name LinkNex-Module5 --icon assets/linknex_app.ico --add-data "assets;assets" --collect-all imageio_ffmpeg --exclude-module open3d --exclude-module torch main.py
```

完整分发 `dist/LinkNex-Module5/` 文件夹。
