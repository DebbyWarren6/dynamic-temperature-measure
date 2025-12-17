# 动态温度测量（OCR + 拆帧 + 汇总绘图）

本仓库用于将实验视频拆成帧图片，并从帧图的指定 ROI 区域 OCR 识别温度（支持多 ROI / 批处理 / 拼接加速 / 失败定位），最终汇总得到“温度-时间”曲线与合并数据表。

> 约定：本项目脚本均为“无 CLI”，所有参数在脚本顶部修改。

---

## 1. 目录结构约定

脚本默认按如下结构工作（你也可以用“遍历所有子文件夹”的模式，见后文）：

- 单次实验目录（示例）：`1217wang/`
- 实验目录下若干“段”目录（示例）：`0.00/`、`0.01/`、`0.02/`…（以前也可能是 `100mW/150mW/...`）
- 每个段目录下包含：
  - `frames/`：帧图片（`.png/.jpg/...`）
  - `frames.csv`：该段 OCR 输出（脚本生成）
  - `temp_vs_time.png`、`combined_temp_vs_time.csv`：汇总脚本生成（在实验目录下）

---

## 2. 环境依赖

### 2.1 Python 依赖

至少需要：

- Python 3.10+（建议）
- `opencv-python`
- `numpy`
- `rapidocr-onnxruntime`
- `pandas`
- `matplotlib`

如果你用 conda 环境（示例环境名 measure），可在该环境里安装：

```bash
pip install opencv-python numpy rapidocr-onnxruntime pandas matplotlib
```

> RapidOCR 默认多为 CPU 推理；如果你希望 GPU 加速，需要额外配置 ONNXRuntime GPU 版本（是否能显著提速取决于模型/分辨率/瓶颈）。

### 2.2 ffmpeg（用于拆帧）

- 需要本机可调用 `ffmpeg`
- 如需 NVIDIA 硬件加速（可选）：确保驱动正常、ffmpeg 支持对应硬件加速

---

## 3. 操作流程（推荐顺序）

### 步骤 A：选 ROI 并保存（只需做一次，ROI 不变就无需重复）

1) 确保你有一个段目录下已经存在 `frames/` 和若干帧图。
2) 运行 ROI 选择脚本：

```bash
python roi_select_and_save.py
```

3) 在弹出的窗口里：

- 在第一张图上框选多个 ROI（例如三行温度区域）
- 确认后会生成/覆盖 `roi_config.json`

输出：

- `roi_config.json`：形如 `{"rois": [[x,y,w,h], ...]}`，并按 y（从上到下）排序。

---

### 步骤 B：把视频按段整理并拆成帧（如果你还没有 frames）

运行：

```bash
python video_power_split.py
```

该脚本会：

- 读取实验目录中的视频
- 按修改时间顺序分配到不同“段”文件夹
- 调用 ffmpeg 拆帧到各段的 `frames/`

> 注意：本脚本已尽量避免插帧/丢帧（例如使用 passthrough / vsync 相关参数），以减少对测量时间轴的影响。

---

### 步骤 C：对整个实验做 OCR（生成每个段的 frames.csv）

运行：

```bash
python roi_temperature_ocr.py
```

在脚本顶部你通常只需要关注这些参数：

- `EXPERIMENT_DIR`：要处理的实验目录（例如 `Path("1217wang")`）
- `FRAMES_SUBDIR`：默认 `"frames"`
- `OUTPUT_CSV_NAME`：默认 `"frames.csv"`
- `ROI_CONFIG`：默认 `roi_config.json`

性能相关（建议从默认开始）：

- `STITCH_ENABLED`：是否拼接 mosaic（减少 OCR 调用次数）
- `STITCH_BATCH_SIZE`：每次 mosaic 处理多少张帧（例如 64/128/256）
- `PREFETCH_ENABLED`：是否开启预读（把 `cv2.imread` 和 OCR 重叠）
- `PREFETCH_WORKERS`：预读线程数（例如 4/8/12）
- `EAGER_PREFETCH_ALL`：是否一次性把该段所有帧都提交预读队列（默认 False）

计时输出（排查瓶颈用）：

- `TIMING_ENABLED`：打印计时
- `TIMING_DETAILED`：打印 imread/crop/mosaic/ocr/fallback 分解

输出：

- 每个段目录下生成 `frames.csv`
- 控制台会打印每个 batch 的 `prep/ocr/total` 以及详细分解（若开启）

常见问题：

- 若报错 `no temperature extracted`：表示该帧在 ROI 内未提取到数字温度；错误信息会包含具体帧文件名，方便定位。

---

### 步骤 D：汇总绘图（生成温度-时间曲线）

运行：

```bash
python aggregate_temp_plot.py
```

脚本会：

- 遍历 `BASE_DIR` 下所有包含 `frames.csv` 的子目录（不再依赖文件夹名必须是 100mW 之类）
- 读取每段的 `frames.csv`
- 对 `temp1/temp2/...` 做聚合（默认可选 mean/median/max）
- 可选尖刺滤波（rolling median + threshold）
- 画出多条曲线并保存图

输出（在实验目录下）：

- `combined_temp_vs_time.csv`
- `temp_vs_time.png`

---

## 4. 速度调参建议（简要）

如果你看到计时里：

- `imread` 占比很高：
  - 开 `PREFETCH_ENABLED=True`
  - 调 `PREFETCH_WORKERS`（先试 8）
  - PNG 解码很重的话，预读通常能带来明显提升
- `ocr_items` 占比很高：
  - 适当增大 `STITCH_BATCH_SIZE`（例如 64→128→256），观察是否更快且稳定
- 某些 batch 出现 `fallback` 很高：
  - 说明有个别帧识别困难，会触发单 tile 回退 OCR

> 建议每次只改一个参数，对比 `total elapsed` 和失败率。

---

## 5. Git 忽略规则说明

本仓库 `.gitignore` 用于忽略大量数据文件（图片/视频/CSV 等）。

注意：`.gitignore` 只对“尚未被 git 跟踪”的文件生效。

如果某些文件之前已经被 `git add/commit` 过，想让它们也被忽略，需要先从索引移除（保留本地文件）：

```bash
git rm -r --cached .
```

然后重新提交一次。

---

## 6. 常用命令（推送到 main）

你已在 `main` 分支时：

```bash
git push -u origin main
```

后续：

```bash
git push
```
