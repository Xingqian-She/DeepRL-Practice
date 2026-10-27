# DeepRL-Practice

深度强化学习算法代码实践与学习笔记。采用 **共享 Docker 环境＋相对独立的实验子项目**：统一环境和实验产物约定，各项目自行组织代码、参数和数学笔记。

当前提供 CartPole 随机策略环境验证，尚未实现训练算法。随机策略的回报仅用于检查运行链路。

## 环境与前提

- Windows Docker Desktop，启用 **WSL2 后端、Linux 容器和 Ubuntu 的 WSL integration**。
- 下文命令从 WSL 终端执行，仓库位于 `/mnt/d/py/DeepRL-Practice`。
- Python、Gymnasium Classic Control、CUDA 版 PyTorch、TensorBoard 和视频依赖都安装在容器内，宿主机无需安装这些 Python 包。
- GPU 需要支持 WSL 的 NVIDIA Windows 驱动；默认运行和录制不需要 GPU。
- 弹窗使用 WSLg。LaTeX 是当前例外：源码放各项目 `theory/`，使用本机已有 LaTeX 环境编译，镜像不包含 TeX。

```bash
cd /mnt/d/py/DeepRL-Practice
docker version
docker compose version
```

仓库先保留在 D 盘；以后大量文件读写确有性能瓶颈时，再考虑迁入 WSL 文件系统。

## VS Code 容器开发

项目提供 `.devcontainer/devcontainer.json`。VS Code 的界面仍在 Windows 上，Python 扩展、Pylance、调试器和终端则运行在开发容器中，使用容器的 `/usr/local/bin/python`。这样补全、导入检查和代码运行使用同一套已安装依赖，宿主机无需再安装 PyTorch 等 Python 库。[VS Code 官方说明](https://code.visualstudio.com/docs/devcontainers/create-dev-container)

此开发配置针对当前 Windows + NVIDIA GPU + WSLg 环境，复用已构建的 `deeprl-practice:py311-cu128` 镜像，不会自动重新下载训练依赖。首次在新机器使用时，先按下一节构建镜像。

WSLg 显示接口只读挂载；编辑器使用独立、可写的容器运行目录 `/run/deeprl`，避免 VS Code 的进程通信 socket 写入只读的宿主机目录。

1. Windows VS Code 安装 **WSL** 和 **Dev Containers** 扩展；本机已安装。
2. 在 **Windows VS Code 用户设置（JSON）**中补充以下两项，保留已有的其他设置；本机已配置。这是应用级设置，不能放到仓库的 `.vscode/settings.json`。仓库位于 D 盘时，这两项确保扩展仍在 WSL 中执行启动命令并读取 WSLg 变量。

   ```json
   {
     "dev.containers.executeInWSL": true,
     "dev.containers.executeInWSLDistro": "Ubuntu-22.04"
   }
   ```

3. 从 **Windows PowerShell** 打开仓库对应的 WSL 窗口：

   ```powershell
   code --new-window --remote wsl+Ubuntu-22.04 /mnt/d/py/DeepRL-Practice
   ```

4. 等待首次 WSL 服务器下载和连接完成。然后按 `Ctrl+Shift+P`，执行 **Dev Containers: Reopen in Container**（在容器中重新打开）。首次连接容器还会下载容器内 VS Code Server 和扩展。
5. 等左下角显示 **Dev Container: DeepRL (GPU + WSLg)**，再打开 Python 文件。容器内会自动安装 Python、Pylance 和 Python Debugger 扩展。

请打开整个仓库根目录，以便发现开发容器配置；子项目仍然可以自行组织代码。仅显示 `WSL: Ubuntu-22.04` 时，编辑器尚未连接到 Docker 容器。此配置依赖 WSLg 环境变量，不要直接从 Windows 本地文件夹执行“在容器中重新打开”；先按上述步骤进入 WSL 窗口。

在 VS Code 的新终端验证：

```bash
which python
python -c 'import sys, torch, gymnasium; print(sys.executable); print(torch.__version__, gymnasium.__version__); print("CUDA:", torch.cuda.is_available())'
python tools/smoke_test.py --mode human
```

预期解释器为 `/usr/local/bin/python`，CUDA 为 `True`，弹窗显示正常。在容器终端里直接执行 `python`，无需再套 `docker compose run`。开发容器未安装 Docker CLI 或挂载 Docker socket；构建镜像、管理 TensorBoard 等 Docker 命令仍在宿主机 WSL 终端执行。

- **断点调试：** 打开目标 `.py`，设置断点，按 F5，选择“Python: 调试当前文件（容器内）”。工作目录设为该文件的目录，适合独立子项目；也可选择专门的 CartPole 弹窗验证配置。
- **导入仍标红：** 确认已连接 Dev Container，等待 Python/Pylance 扩展安装完成，再执行 **Python: Select Interpreter**，选择 `/usr/local/bin/python`。必要时执行 **Python: Restart Language Server**。
- **更新依赖：** 修改根目录依赖、生成锁并重建共享镜像后，执行 **Dev Containers: Rebuild Container**。仅修改 Python 源码无需重建。
- **结束开发：** 关闭该 VS Code 远程窗口会停止开发容器，宿主机源码和 `runs/` 输出仍保留；独立的 TensorBoard 服务不受影响。

LaTeX 继续按约定在本机编译。编辑推导时可另开一个普通 Windows VS Code 窗口使用本机 LaTeX Workshop；开发容器不安装 TeX。

## 构建与运行

以下命令在仓库根目录执行。首次构建会下载较大的 CUDA/PyTorch 依赖。

```bash
docker compose build runner
docker compose run --rm runner python -m pip check
docker compose run --rm runner python -c 'import gymnasium, torch, tensorboard; print(gymnasium.__version__, torch.__version__, tensorboard.__version__)'

# 默认：不渲染、3 回合、seed=42，不申请 GPU、不挂载 WSLg
docker compose run --rm runner python tools/smoke_test.py

# 不弹窗，每回合输出一个 MP4
docker compose run --rm runner python tools/smoke_test.py --mode video

# output-dir 是父目录，每次在其中创建唯一运行目录
docker compose run --rm runner python tools/smoke_test.py \
  --mode none --episodes 5 --seed 7 --output-dir tools/runs
```

默认使用官方 PyPI。若下载缓慢，可选用镜像源构建，锁定版本保持一致：

```bash
docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple runner
```

默认输出保存在 `tools/runs/`，容器退出后仍保留在宿主机。脚本仅使用随机动作，不保存模型。指标 `episode/return` 和 `episode/length` 的横轴为累计环境步数；遇到 `terminated` 或 `truncated` 均结束当前回合。

### GPU

GPU 配置只作用于 `runner`。以下检查明确要求 CUDA，不会静默回退 CPU：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm runner \
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(3, device="cuda"); print(torch.cuda.get_device_name(0), x.sum().item())'
```

随机策略没有神经网络，动作采样仍在 CPU 执行；CUDA 张量检查单独验证 GPU 链路。未来训练脚本可自行提供 `--device` 参数。

### WSLg 弹窗

在 **WSL 终端** 检查显示变量和接口，再启动窗口：

```bash
echo "$DISPLAY / $WAYLAND_DISPLAY / $XDG_RUNTIME_DIR"
ls -ld /tmp/.X11-unix /mnt/wslg "$XDG_RUNTIME_DIR"
docker compose -f compose.yaml -f compose.gui.yaml run --rm runner \
  python tools/smoke_test.py --mode human
```

窗口在 Windows 桌面显示，运行结束后关闭。不要在 PowerShell 直接执行 GUI 命令。显示变量缺失时 Compose 明确报错；接口目录缺失也会报错，不自动创建空目录。

CartPole 使用 SDL/X11，GUI 配置强制选择 X11 并挂载 WSLg 接口。当前 CartPole 验证将音频驱动设为 `dummy`，避免探测不存在的声卡；需要声音的游戏应另行配置音频。其他游戏的 OpenGL 加速可能需要额外设置，须独立验证；CUDA 运算成功并不代表 OpenGL 已配置好。

GPU 与 GUI 可组合：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml -f compose.gui.yaml run --rm runner \
  python tools/smoke_test.py --mode human
```

### TensorBoard

```bash
docker compose --profile monitoring up -d tensorboard
docker compose logs tensorboard
```

在 Windows 浏览器打开 [http://localhost:6006](http://localhost:6006)，查看 Scalars。服务只读挂载仓库，递归发现各项目事件文件，各次运行分别显示。没有数据时先执行一次验证脚本。

```bash
# 停止本项目服务；不会删除宿主机输出
docker compose --profile monitoring down
```

## 子项目组织

```text
DeepRL-Practice/
├── Dockerfile
├── compose.yaml                  # 默认环境＋按需启动的 TensorBoard
├── compose.gpu.yaml              # 可选 NVIDIA GPU
├── compose.gui.yaml              # 可选 WSLg 显示接口
├── requirements.in               # 直接依赖
├── requirements.lock             # Linux / Python 3.11 完整版本锁
├── tools/
│   ├── smoke_test.py
│   └── runs/                     # 自动生成，不提交
└── <游戏名>-<算法名>/             # 实际学习时再创建
    ├── README.md
    ├── train.py                  # 推荐命名，非强制接口
    ├── evaluate.py
    ├── ...                       # 内部结构自由组织
    ├── theory/                   # 可选：LaTeX、推导、参考资料
    └── runs/                     # 自动生成，不提交
```

- 目录使用小写名称，如 `cartpole-dqn`；README 和运行记录写准确的环境 ID，如 `CartPole-v1`。
- 各项目从自己的目录直接运行，不导入其他子项目，也不依赖根目录修改 `PYTHONPATH`。不建立公共算法框架。
- 简单项目可以只有一个脚本；复杂项目自行拆分网络、算法、回放池和配置。
- 首次学习算法时可在 `theory/` 推导，后续项目直接链接已有推导。LaTeX 中间文件忽略，源码和精选 PDF 可提交。
- 项目 README 至少说明目标、环境 ID、运行命令、重要参数、结果和参考资料；未验证的结果标明待验证。

### 新增项目

后续开始学习 DQN 时自行创建目录并编写脚本；首轮不创建空算法项目：

```bash
mkdir -p cartpole-dqn
# 在 cartpole-dqn 内编写 README.md、train.py、evaluate.py 等

# 脚本创建后，从该项目目录运行；源码修改无需重建镜像
docker compose run --rm -w /workspace/cartpole-dqn runner python train.py
docker compose run --rm -w /workspace/cartpole-dqn runner python evaluate.py

# 交互式开发
docker compose run --rm -w /workspace/cartpole-dqn runner bash
```

这些入口由项目自行实现。建议训练默认不渲染，单独加载模型评估时选择弹窗或录制。需要 GPU 或弹窗时，叠加相应 `-f` 配置。

### 实验产物

各项目在内部保存独立运行目录：

```text
runs/<UTC时间戳-seed-唯一后缀>/
├── metadata.json                 # 参数、环境 ID、依赖、设备、Git 状态
├── tensorboard/
├── checkpoints/                  # 有模型时才生成
└── videos/                       # 有录制时才生成
```

验证脚本遵循该约定，仅创建需要的文件；元数据还记录每回合结果、完成状态和失败信息。Git 不可读取时记录 `null`，不假定工作区干净。设置种子有助于重复检查，但不保证未来 GPU 训练跨硬件完全一致。

所有 `runs/` 均被 Git 和 Docker 构建上下文忽略。精选结果可另存项目 `results/` 并明确提交。共享镜像不包含源码或实验产物，它们在运行时挂载。

## 更新依赖

共享环境首轮只包含 Classic Control，不预装 Box2D、Atari、MuJoCo。新增依赖统一修改 `requirements.in`，重新生成锁并构建；不要依赖给运行中的容器临时 `pip install` 来维护环境。

TensorBoard 2.20 的启动入口仍使用 `pkg_resources`，因此直接依赖中固定了兼容的 `setuptools==80.9.0`。镜像构建会执行 `pip check` 和 `tensorboard --version`，运行服务后还需检查页面是否可访问。

以下命令用于主动更新依赖，会覆盖锁文件：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace python:3.11.14-slim-bookworm \
  sh -c 'pip install --no-cache-dir uv==0.9.5 && uv pip compile requirements.in --output-file requirements.lock --python-version 3.11 --python-platform x86_64-unknown-linux-gnu'

docker compose build runner
docker compose run --rm runner python -m pip check
docker compose run --rm runner python tools/smoke_test.py
docker compose run --rm runner python tools/smoke_test.py --mode video
```

构建使用 BuildKit 的 uv 下载缓存。若仅小幅调整依赖、Python 和基础系统版本保持不变，也可以复用本机已有的本项目镜像，避免重新下载已安装的 CUDA 库：

```bash
docker compose build \
  --build-arg BASE_IMAGE=deeprl-practice:py311-cu128 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple runner
```

首次构建和升级 Python／基础系统时使用默认基础镜像，不传 `BASE_IMAGE`。

锁针对 Linux x86_64 / Python 3.11；PyTorch 2.9.1 的 PyPI Linux wheel 带有 CUDA 12.8 支持（安装版本标识为 `2.9.1`，运行时可显示 `2.9.1+cu128`）。镜像内安装 CUDA 运行库，宿主机使用 NVIDIA Windows 驱动。更新 PyTorch/CUDA 后须重新做 GPU 检查。系统库在 Dockerfile 维护，基础镜像固定版本标签；字节级复现还需要固定镜像 digest 和系统软件源快照。

## 验收

分别验证构建与依赖、无窗口运行、可解码视频、多个 run 的 TensorBoard 指标、真实 GPU 张量运算、桌面窗口，以及退出后输出持久化。各项实际结果见 [VALIDATION.md](VALIDATION.md)；配置文件存在不代表链路已验证。

参考：[Docker WSL2](https://docs.docker.com/desktop/features/wsl/)、[Compose GPU](https://docs.docker.com/compose/how-tos/gpu-support/)、[WSLg 容器](https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md)、[Gymnasium 录制](https://gymnasium.farama.org/introduction/record_agent/)。
