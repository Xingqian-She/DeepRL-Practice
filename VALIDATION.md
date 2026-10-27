# 基础环境验收记录

日期：2026-10-26。以下区分静态检查和真实运行，不将配置存在视为验收通过。

## 环境

- Windows Docker Desktop 4.96.3，Engine 29.9.2，Linux 容器。
- Ubuntu-22.04 / WSL2；WSLg 显示变量、`/mnt/wslg` 和 X11 接口存在。
- NVIDIA GeForce RTX 4090 Laptop GPU；Windows 驱动 616.87。
- 实测环境：Python 3.11.14、PyTorch 2.9.1+cu128（包元数据版本为 2.9.1）、Gymnasium 1.2.2、TensorBoard 2.20.0、setuptools 80.9.0。
- 镜像：`deeprl-practice:py311-cu128`；最终镜像 ID：`sha256:880a831da24939d9645555b3dce392d250c2502b876095eb17d37571062c13a1`。

## 当前结果

| 检查 | 状态 | 证据或限制 |
|---|---|---|
| Python 语法、Git diff 空白检查 | 通过 | AST 解析、`git diff --check` |
| 默认 / GPU Compose 配置 | 通过 | `docker compose config --quiet` |
| GUI / GPU+GUI 配置 | 通过 | 从 Ubuntu WSL 分别运行 human 模式 3 回合、GPU+GUI 组合 10 回合，均正常完成 |
| 缺失显示变量报错 | 通过 | 未设置 DISPLAY 等变量时，GUI 配置明确拒绝启动 |
| 实验产物忽略规则 | 通过 | `git check-ignore tools/runs/verify_artifacts.py` |
| 完整依赖锁 | 通过 | Linux Python 3.11 中解析，并逐项核对全部 47 个已安装版本 |
| 镜像构建 / pip check | 通过 | 镜像导出成功，无损坏依赖；`tensorboard --version` 返回 2.20.0 |
| 默认无窗口运行 | 通过 | 不挂载 WSLg、不申请 GPU；CUDA 不可见，完成 3 个随机策略回合 |
| 重复运行 / 记录 | 通过 | 同种子两次运行结果相同，创建不同目录；元数据、事件文件和 Git 状态可读取 |
| 无窗口视频 | 通过 | 三个 600×400 MP4，帧数为 31、94、45；ffprobe 计帧和 ffmpeg 全部解码成功 |
| TensorBoard 指标文件 | 通过 | 两个标量各有 3 条记录，数值及环境步数与回合结果一致 |
| TensorBoard 服务 | 通过 | Windows 宿主机访问 localhost:6006 返回 HTTP 200，多 run 的指标 API 正常；挂载 RW=false |
| GPU | 通过 | CUDA 12.8 可见，识别 RTX 4090 Laptop；GPU 张量求和结果为 3.0 |
| 输出持久化 | 通过 | 一次性容器删除后，宿主机仍有 metadata.json、事件文件和三个 MP4 |
| 参数错误 | 通过 | episodes=0、seed=-1、缺少 DISPLAY 的 human 模式均返回 2，未创建运行目录 |
| 异常和中断状态 | 通过 | 注入 step 异常记录 failed；注入 KeyboardInterrupt 记录 interrupted；均写入结束时间 |
| WSLg 桌面弹窗 | 通过 | 连接 WSLg X11，3 回合完成且退出码为 0；用户确认 Windows 桌面窗口显示正常 |

## 运行证据与构建说明

最终镜像验证中，seed=42 的三个回合回报和长度均为 `30, 93, 44`；无窗口重复运行与视频模式结果一致。回报曲线的横轴为累计环境步数 `30, 123, 167`。

最终验证的正常运行目录如下，均位于宿主机 `tools/runs/acceptance/` 下且不提交 Git：

```text
20261026T113543194828Z-seed42-5e7db68e  # 无窗口第一次
20261026T113545849980Z-seed42-5a50b23f  # 无窗口第二次
20261026T113548439601Z-seed42-9a05852b  # 视频
```

TensorBoard 服务 API 实际读取了第一组运行的 `episode/return` 和 `episode/length`，两者都返回 `30,93,44`。验收结束时服务保持启动，可直接访问 [http://localhost:6006](http://localhost:6006)；停止命令见 README。

先从固定 Python 基础镜像完整构建，再通过 `BASE_IMAGE` 复用同版本项目镜像更新 setuptools。最终容器中所有 47 个版本均与当前锁一致，CPU、GPU、日志和视频验收已在兼容性修复后重新执行。构建使用可配置的清华 PyPI 镜像源，默认配置仍为官方 PyPI。

服务启动检查发现 TensorBoard 2.20 仍导入 `pkg_resources`，而 setuptools 新版已移除该模块，故显式固定 `setuptools==80.9.0`；构建也加入了 CLI 启动检查。[setuptools 的移除说明](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html)。TensorBoard 提示未安装 TensorFlow，但本项目的 PyTorch 标量记录及读取已实测正常。

## WSLg 实测

用户已在 Docker Desktop → Settings → Resources → WSL Integration 中单独启用 `Ubuntu-22.04`。默认发行版集成开关保持关闭也可运行。WSL 内现已能够访问 Docker Engine 29.9.2，显示变量为 `DISPLAY=:0`、`XDG_RUNTIME_DIR=/run/user/1000`。

实际执行命令：

```bash
docker compose -f compose.yaml -f compose.gui.yaml run --rm -T runner \
  python tools/smoke_test.py --mode human

docker compose -f compose.yaml -f compose.gpu.yaml -f compose.gui.yaml run --rm -T runner \
  python tools/smoke_test.py --mode human --episodes 10
```

第一组完成 3 回合，回报为 `30,93,44`，用户确认看到了正常显示的 CartPole 窗口。第二组完成 10 回合，运行记录中的 `cuda_available=true`；随机动作策略本身仍使用 CPU。两组运行均正常退出，元数据状态为 `completed`。

```text
tools/runs/20261026T120807190181Z-seed42-75e2386b  # GUI
tools/runs/20261026T120917536340Z-seed42-51cd41ad  # GPU+GUI
```

首次 GUI 运行有音频设备探测提示。CartPole 无音频需求，故 GUI 配置增加 `SDL_AUDIODRIVER=dummy`；第二组实测已无 ALSA 和音频运行目录所有权提示。仍可出现上游 `pkg_resources` 弃用提示，不影响已验证的窗口运行。

## VS Code 开发容器

- 增加 `.devcontainer/devcontainer.json`，复用同一训练镜像，工作目录为 `/workspace`；自动配置 Python、Pylance、Python Debugger 和 `/usr/local/bin/python`。
- 增加当前 Python 文件和 CartPole 弹窗的 F5 调试配置；当前文件的工作目录为其所在目录。
- 使用已安装扩展自带的 Dev Containers CLI 0.89.0，在 Ubuntu WSL 中真实启动配置，结果为 `outcome=success`，`postCreateCommand` 的 `pip check` 通过。
- 在该验收容器中实测解释器路径正确，PyTorch、Gymnasium、TensorBoard 可导入；CUDA 可见，GPU 张量求和为 3.0；human 模式完成 3 回合，回报为 `30,93,44`，元数据状态为 `completed`。
- 首次 VS Code 界面连接发现 D 盘目录被按 Windows 路径启动，导致缺失 WSLg 变量。已备份原 Windows VS Code 用户设置，并仅新增 `dev.containers.executeInWSL=true`、`dev.containers.executeInWSLDistro=Ubuntu-22.04`，让界面采用已验证的 WSL 执行路径。
- 修正启动设置后，VS Code 界面已连接至 `DeepRL (GPU + WSLg)`，`postCreateCommand` 完成；实际界面容器中解释器为 `/usr/local/bin/python`，核心库均可导入，CUDA 为 True。Python、Pylance、Python Debugger 均已安装，用户截图确认 F5 使用容器内 Python Debugger 启动了脚本。
- 收尾日志发现 VS Code Agent IPC 试图写入只读的宿主机运行目录。配置已增加权限 0700 的私有 tmpfs `/run/deeprl`，Wayland 使用宿主机 socket 的绝对路径。新版配置再次通过 CLI 启动、目录权限及 Unix socket 创建/删除检查，并完成 1 回合 human 模式。
- 私有运行目录修正后，用户已执行 Rebuild Container 并重新连接。实际界面容器中的 `/run/deeprl` 可创建和删除 Unix socket；VS Code 已在其中建立 IPC/Git socket，新日志中无 EROFS 或 Agent Host 启动失败。解释器与 Machine Settings 均为 `/usr/local/bin/python`，CUDA 为 True；Python、Pylance、Python Debugger 安装成功。Pylance 日志确认使用 Python 3.11.14、完成索引并以容器解释器执行路径解析。
