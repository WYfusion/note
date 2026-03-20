# Epoch 026 验证集混淆矩阵分析

## 1. 整体指标快照
- **覆盖物种数**：179（验证集样本数 300,344 条）
- **宏平均 Recall**：64.5%；**微平均准确率**：66.8%
- **Recall 四分位**：P25=56.8%，中位=67.9%，P75=77.5%
- **样本数分布**：最少 42（`Himalayan_Swiftlet`），最多 4,250（长尾差 >100×）；P10=791，P90=2,693 条
- **Recall vs. log(样本数) 皮尔逊相关**：0.38 —— 长尾效应明显，但并非唯一原因
- **Precision 行**显示个别类别出现 `nan` 或异常高值，暗示预测端仍有偏置；但本报告聚焦于数据侧可控的改进

## 2. Recall < 40% 的物种（17 个）
| 物种                       |   样本数 | Recall | 最高混淆方向                                                                                     | 判定            | e+n      |
| ------------------------ | ----: | -----: | ------------------------------------------------------------------------------------------ | ------------- | -------- |
| Himalayan_Swiftlet       |    42 |   0.0% | Pied_Kingfisher (31.0%), Buff-bellied_Pipit (19.0%), Common_Kingfisher (14.3%)             | 数量太少了，无法学习到   | 9+7      |
| Falcated_Duck            | 1,547 |   2.0% | Gadwall (37.6%), Mallard (17.2%), Eastern_Spot-billed_Duck (7.1%)                          | 有点少           | 4+16     |
| Yellow_Bittern           |   714 |   5.3% | Greater_Coucal (7.1%), Large_Hawk-_Cuckoo (6.4%), Black-crowned_Night_Heron (5.7%)         | 品质太差了         |          |
| Eastern_Spot-billed_Duck |   938 |   6.6% | Mallard (30.6%), Light-vented_Bulbul (11.3%), Red_Collared_Dove (4.6%)                     | 筛选设置有一定差池     |          |
| Swinhoe's_Snipe          |   518 |  11.0% | Whiskered_Tern (52.7%), Pin-tailed_Snipe (9.3%), Common_Snipe (4.4%)                       | 剔除部分，但和WT确实很像 |          |
| Pacific_Reef_Heron       |   231 |  14.3% | Great_Egret (23.4%), Large-billed_Crow (13.4%), Chinese_Pond_Heron (8.7%)                  | 太少了           | 9+6      |
| Greater_Sand_Plover      |   875 |  28.9% | Lesser_Sand_Plover (14.9%), White-winged_Tern (7.9%), Curlew_Sandpiper (4.6%)              | 剔除了部分         |          |
| Crested_Myna             | 1,820 |  30.1% | Great_Egret (4.5%), Eurasian_Tree_Sparrow (4.3%), Black-collared_Starling (4.0%)           | 剔除了部分(有点少)    |          |
| Sooty-headed_Bulbul      | 1,868 |  30.4% | Red-whiskered_Bulbul (10.2%), Light-vented_Bulbul (4.7%), Common_Redshank (2.9%)           | 剔除了部分         |          |
| Eastern_Imperial_Eagle   |   784 |  31.4% | Common_Snipe (17.7%), Northern_Shoveler (7.3%), Great_Crested_Grebe (7.1%)                 | 有点少           | 15+11/16 |
| Chestnut-eared_Bunting   | 1,071 |  32.2% | Siberian_Rubythroat (10.3%), Little_Bunting (7.7%), Black-browed_Reed_Warbler (5.6%)       | 剔除了部分         |          |
| Scaly-breasted_Munia     | 1,666 |  33.1% | Asian_Brown_Flycatcher (5.5%), Pallas's_Leaf_Warbler (3.8%), Eastern_Yellow_Wagtail (3.4%) | 剔除了部分         |          |
| White-rumped_Munia       | 1,652 |  34.9% | Black-browed_Reed_Warbler (11.6%), Curlew_Sandpiper (7.3%), Pale_Thrush (3.6%)             | 剔除了部分         |          |
| Stejneger's_Stonechat    |   476 |  35.1% | Siberian_Rubythroat (38.7%), Blue_Rock_Thrush (3.2%), Verditer_Flycatcher (3.2%)           |               |          |
| Chinese_Grey_Shrike      |   693 |  37.5% | Common_Kingfisher (11.7%), Garganey (8.1%), Pacific_Golden_Plover (6.3%)                   | 数量太少了         | 16+11/22 |
| Asian_Brown_Flycatcher   | 1,744 |  37.7% | Grey-backed_Thrush (8.0%), Grey_Wagtail (6.5%), Yellow-browed_Warbler (4.4%)               | 剔除了部分         |          |
| Tristram's_Bunting       | 1,722 |  37.9% | Pallas's_Leaf_Warbler (18.5%), Siberian_Rubythroat (5.8%), Dusky_Warbler (3.4%)            | 剔除了部分         |          |


> **优先级提示**：粗体样本数 >1,500 但 recall <40% 的类（Falcated_Duck、Crested_Myna、Sooty-headed_Bulbul、Scaly-/White-rumped Munia、Asian_Brown_Flycatcher、Tristram's_Bunting）最值得首先返工，它们证明“量足但质不佳”。

## 3. 数据侧主要症状
1. **家族内声学重叠 + 标签漂移**  
   - 水鸟类（Falcated/Eastern Spot-billed Duck、Swinhoe's Snipe、Greater Sand Plover）相互混淆在情理之中，但出现向雀形目（Light-vented Bulbul）或鸽类（Red Collared Dove）的跳跃，说明目录内混入了完全不同物种的录音。  
   - 林地小型 Passerines（Chestnut-eared/Tristram's bunting、Scaly-/White-rumped Munia、Asian Brown Flycatcher）互相预测，且被 Siberian Rubythroat、Grey-backed Thrush 吸走，极可能是分段时间包含多种叫声但未做多标签标注。

2. **长尾极端 + 录音稀薄**  
   - `Himalayan_Swiftlet` 只有 42 段且全数跑偏，说明录音极短或背景噪声抹平了特征。该类建议暂时并入“其他 swiftlet”或补采数据，否则会继续拖累损失。  
   - `Pacific_Reef_Heron`、`Stejneger's_Stonechat` 等 <500 条的类别也很难仅靠建模弥补。

3. **噪声 / 空段 / 重复波形**  
   - `Crested_Myna` 与白鹭、树麻雀互混，常见于“巢区环境录音”——背景里有多种鸟但文件按照文件夹名统统打上了 Myna。  
   - `Sooty-headed_Bulbul` 出现 2.9% 的 Common_Redshank，说明文件中可能包含河滩环境的远距离叫声甚至浪花。

4. **切分策略引入的系统误差**  
   - `Swinhoe's_Snipe` 大量被识别成 `Whiskered_Tern`，对应 `split_segment.py` 中用统一阈值切分所有水鸟的策略；当背景是机场/池塘的连续噪声时，频域模式更像燕鸥。  
   - `train_vad_segment_summary.jsonl` 中的能量统计如果未过滤，将导致静音段被当作负样本塞进正类目录。

## 4. 数据优化路线（按操作颗粒度划分）
### 4.1 Tier 0 — 自动统计 & 可视化（当天可完成）
- **脚本扫描**：
  - 统计每个物种文件夹的有效时长、平均 SNR、峰值频段（`librosa.feature.melspectrogram`）。
  - 生成“候选噪声清单”（如均值能量 < -40 dB 的文件）并导出 CSV 供人工复核。
- **交叉验证**：把上表的 17 类写入 `data/review_targets.txt`，训练/验证列表生成脚本优先抽样这些类做人工 spot check。

### 4.2 Tier 1 — 手动复核（需要耳听 + 快速标记）
1. **基于文件夹的抽检**（结合你现有“按物种建文件夹”的结构）：
   - 为每个低 recall 类创建 `species/review/` 子文件夹；把可疑片段（背景中出现其他物种或静音）移动进去，暂时从训练集中排除。  
   - 建议每个物种至少抽听 5×1min；可用 `ffplay -nodisp -autoexit path.wav` 搭配 `tmux` 批量播放。
2. **多标签补注**：对确认“多鸟同录”的文件，在 `train_list.txt` 中重复登记（同路径但不同标签），同时把辅助标签权重设为 <1（在 loss 端实现），避免一刀删掉珍贵素材。
3. **错放目录纠正**：
   - 重点排查 `Falcated_Duck`、`Eastern_Spot-billed_Duck` 文件夹中是否混入“岸边环境”录音：若出现大量麻雀/鹭叫声，直接移动到对应物种目录。
   - 对 `Crested_Myna`、`Sooty-headed_Bulbul` 这类城市留鸟，建议建立 `ambient` 子目录，把包含明显人声/交通噪声的文件迁出并做降噪或静音裁剪。

### 4.3 Tier 2 — 结构性调整（配合下一轮训练）
- **重构 split / mix 策略**：对水鸟/涉禽单独设置 VAD 阈值与最短段长度；或直接改为“整段 + 淡入淡出”而不做细粒度切片，避免把单调背景误当成特征。  
- **硬样本再权重**：把上述 17 类写入 `hard_species.txt`，训练时对其采样概率乘以 2~3，并使用双重 mixup（positive-only + negative-only）增加判别性。  
- **噪声库扩展**：收集同一栖息地的纯背景噪声，加入 `nodenoise_nosilence_mix`，在 `preprocessing/` 流程里随机混合，以免模型把环境声作为标签。  
- **校验文件清单**：
  - 重新生成 `label_list.txt` 与实际文件夹同步；  
  - 用 `source_based_split.py` 保证 train/val/test 不共享同一录音源，防止标签污染再次混入验证集。

## 5. 快速执行清单
| 步骤 | 目标 | 操作建议 |
| --- | --- | --- |
| 1 | 锁定问题物种 | 将上表 17 类写入 `report/review_targets.yaml`（自建），供脚本提取。 |
| 2 | 批量音频 QA | 在项目根运行 `python util.py --scan-folders data/audio --targets report/review_targets.yaml --min-duration 3`，导出可疑文件表（可自行实现）。 |
| 3 | 手动复核 | 依次进入每个物种目录，创建 `review/` 与 `keep/` 子目录，通过听音/谱图（`librosa.display.specshow`）决定去留。 |
| 4 | 清洗/补标 | - 对误标文件重命名并移入正确物种；  
- 对多鸟文件在 `train_list.txt` 追加一行副标签；  
- 对静音/噪声段直接删除或转入噪声库。 |
| 5 | 重新生成列表 | 运行你已有的列表脚本（如 `data/source_based_split.py`）确保新的 train/val 划分同步更新。 |
| 6 | 针对性回测 | 先只用清洗过的 17 类做快速微调（可缩小 batch/epoch），确认 Recall 抬升后再全量训练。 |

## 6. 后续验证建议
- **可视化监控**：把每轮训练的 per-class recall 写入 `report/per_class_metrics.csv`，用折线图观察数据清洗效果。  
- **分群评估**：按生态位（涉禽/林地/城市/夜行）聚合 Recall，优先攻克指标最差的群组。  
- **对照试验**：对低 Recall 类建立“旧数据 vs. 清洗数据”的听觉对照集，确认误差确实来自标签，而非模型结构不足。

> 当你按物种文件夹管理音频时，**“不要轻易删除，先迁出 /review、/ambient 子目录”** 是最高效的策略：既保持原始素材可回溯，又方便你逐步把干净的数据迁回主目录，确保下一次训练只读取可信片段。
