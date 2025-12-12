# Repository Guidelines

## 项目结构与模块组织
- 代码集中在 `src/`，按功能分包（示例：`src/data/`、`src/models/`、`src/pipelines/`）；入口脚本建议放 `src/main.py` 或 `src/cli.py`。
- 测试放 `tests/` 并镜像源代码路径（例：`tests/data/test_loader.py` 对应 `src/data/loader.py`）；共享夹具放 `tests/conftest.py`。
- 数据与产物：原始数据放 `data/raw/`，处理后数据放 `data/processed/`，模型或权重放 `checkpoints/`，大文件保持 git 忽略。
- 文档放 `docs/`，快速记录放 `docs/notes/`，文件名带日期如 `2024-12-architecture.md`；脚本或一次性工具放 `scripts/`。
- 数据集放在 `dataset/110_video_frames_2025_11_all_200_have_json` 目录中。

## 构建、测试与开发命令
- 使用默认的虚拟环境(base)：conda activate。
- 安装依赖（待有 `pyproject.toml` 或 `requirements.txt`）：`pip install -e .[dev]` 或 `pip install -r requirements.txt`。
- 运行测试：`pytest`。
- 静态检查/格式化（配置好后）：`ruff check .` 与 `ruff format .`。
- 示例运行（按约定添加脚本后）：`python -m src.main --help` 或 `python scripts/train.py --config configs/default.yaml`。

## 代码风格与命名
- 遵循 PEP 8；4 空格缩进，函数/变量用 snake_case，类用 PascalCase。
- 公共接口必须有类型注解与简短 docstring；必要时开启 `from __future__ import annotations`。
- 函数保持精简，复用逻辑拆到小函数/模块，减少副作用。
- 文件读写使用 `pathlib.Path`，避免硬编码；配置集中在 `configs/`（提供 `configs/example.yaml`）。
- 数据变换优先纯函数；有副作用需在 docstring 说明。

## 测试规范
- 使用 `pytest`，测试名描述性（如 `test_loads_small_frame_batch`）；公共夹具放 `tests/conftest.py`。
- 追求快速、可重复；对 I/O 与网络调用使用 mock。
- 每个缺陷修复补回归测试，覆盖边界场景（空输入、损坏帧、极端尺寸）。
- 覆盖关注核心路径（解析、预处理、模型推理），而非仅行覆盖率。

## 提交与 PR 规范
- 提交信息用祈使句，必要时带短前缀（例：`data: fix frame loader edge cases`）。
- 保持单一目的提交；非必要不将重构与功能混合。
- PR 需写变更摘要、原因、测试情况；关联 issue/实验，附重要日志或指标。
- 行为变化需说明前后对比；用户可见变更附截图或样例输出。

## 安全与配置
- 不要提交密钥/证书；使用环境变量，`.env` 保持不入库（已在忽略列表）。
- 验证并清理外部输入文件；格式异常时应拒绝并明确报错。
- 添加配置时提供 `.env.example` 并注明键用途，避免把默认写死进代码。
- 数据、模型权重与中间产物保持在本地或受控存储；必要时用符号链接，勿直接入库。

## 回答说明
所有回答都用中文回复。
