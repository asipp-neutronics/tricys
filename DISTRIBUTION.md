# Offline Distribution Guide

This guide explains how to package the Tricys project and its dependencies for installation on a machine without internet access.

## 1. Prerequisites

- The **Source Machine** (your computer) must have internet access and Python installed.
- The **Target Machine** (other person's computer) needs Python installed.
    - **IMPORTANT**: The Python version on the target machine must roughly match the source machine (e.g., both use Python 3.10).

## 2. Generating the Offline Package

Run the following script on the **Source Machine**:

```bash
script\offline_distribution\generate_offline_package.bat
```

This will:
1.  Read `requirements.txt`.
2.  Download all necessary `.whl` and `.tar.gz` packages into `dist_tricys\packages`.
3.  **Clone** the latest project source code from GitHub (`https://github.com/asipp-neutronics/tricys.git`) to `dist_tricys\src`.
4.  Generate `dist_tricys\install_offline.bat` and `dist_tricys\offline_readme.txt`.

The output will be a folder named `dist_tricys` in the project root.

## 3. Transferring to Target Machine

1.  Copy the entire `dist_tricys` folder to a USB drive or other transfer medium.
2.  Paste it onto the **Target Machine**.

## 4. Installing on Target Machine

1.  Open the `dist_tricys` folder on the target machine.
2.  Right-click `install_offline.bat` and select **Run as Administrator** (recommended, though double-clicking is usually fine if permissions allow).
3.  Follow the on-screen prompts.

The script will:
- Check the Python version.
- Install all dependencies from the local `packages` folder.
- Install the Tricys project itself.

## Troubleshooting

- **Platform Mismatch**: If you generate the package on Windows, it will likely only work on Windows Target Machines because some libraries (`pywin32`, `numpy`, etc.) have OS-specific binary wheels.
- **Python Version Mismatch**: If the installer complains about missing versions, ensure the target machine has the same major.minor Python version as the source.
