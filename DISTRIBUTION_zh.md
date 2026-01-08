# 离线分发指南 (Offline Distribution Guide)

本指南介绍如何将 Tricys 项目及其依赖项打包，以便在没有互联网访问的计算机上进行安装。

## 1. 前提条件

- **源计算机** (您的电脑): 必须具备互联网访问权限并已安装 Python。
- **目标计算机** (他人的电脑): 需要安装 Python。
    - **重要**: 目标计算机上的 Python 版本应与源计算机大致匹配 (例如，两者都使用 Python 3.10)。

## 2. 生成离线包

在 **源计算机** 上运行以下脚本：

```bash
script\offline_distribution\generate_offline_package.bat
```

该脚本将执行以下操作：
1.  读取 `requirements.txt`。
2.  将所有必要的 `.whl` 和 `.tar.gz` 包下载到 `dist_tricys\packages` 目录中。
3.  **克隆 (Clone)** GitHub 上最新的项目源代码 (`https://github.com/asipp-neutronics/tricys.git`) 到 `dist_tricys\src` 目录。
4.  生成 `dist_tricys\install_offline.bat` 和 `dist_tricys\offline_readme.txt`。

输出结果将是项目根目录下一个名为 `dist_tricys` 的文件夹。

## 3. 传输到目标计算机

1.  将整个 `dist_tricys` 文件夹复制到 U 盘或其他传输介质。
2.  将其粘贴到 **目标计算机** 上。

## 4. 在目标计算机上安装

1.  在目标计算机上打开 `dist_tricys` 文件夹。
2.  右键点击 `install_offline.bat` 并选择 **以管理员身份运行** (推荐; 如果权限允许，直接双击通常也可以)。
3.  按照屏幕上的提示操作。

脚本将执行以下操作：
- 检查 Python 版本。
- 从本地 `packages` 文件夹安装所有依赖项。
- 安装 Tricys 项目本身。

## 故障排除

- **平台不匹配**: 如果您在 Windows 上生成该包，它可能仅适用于 Windows 目标计算机，因为某些库 (`pywin32`, `numpy` 等) 具有特定于操作系统的二进制文件。
- **Python 版本不匹配**: 如果安装程序提示缺少版本，请确保目标计算机的主版本号和次版本号 (Major.Minor) 与源计算机一致。
