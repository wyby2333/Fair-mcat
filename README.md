# Fair-mcat

Fair-mcat 的精简源代码。程序采用单一路径完成数据生成、模型训练、候选搜索、候选筛选与最终评估。

## 环境

- Python 3.11
- PyTorch 2.14.0
- 其余版本见 `requirements.txt`

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如需 CUDA，请根据本机 CUDA 环境安装对应的 PyTorch 版本，再安装其余依赖。

## 运行

代码包仅附带 Email 原始图。先生成训练数据：

```bash
python -m data.generate --dataset email --seed_num 10
```

再执行完整实验：

```bash
python gxltd.py --dataset email --seed_num 10
```

控制台显示 Fair-mcat 的运行进度与结果，并隐藏 Degree、PageRank 基线行。

命令参数：

```bash
python -m data.generate --help
python gxltd.py --help
```

生成数据、检查点与结果默认写入 `data/cache/` 和 `results/`，并由 Git 忽略。

## 目录

- `gxltd.py`：程序入口
- `main/`：模型、训练、搜索与评估
- `data/`：Email 图数据及数据处理
- `configs/datasets.json`：运行配置
- `requirements.txt`：Python 依赖
