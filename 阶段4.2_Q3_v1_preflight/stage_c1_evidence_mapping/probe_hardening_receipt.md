# C-1 tokenizer probe 加固回执

日期：2026-09-25

## 修改

原 probe 在检查原始数组之前执行 `np.asarray(text_bert, dtype=np.int64)`，会把小数值截成整数。本次改为先读取原始 dtype 和数值，然后按顺序验证：

1. shape 必须为 3×50，且 dtype 为实数数值类型；
2. 全部有限，浮点值必须精确为整数；token IDs 必须落在 tokenizer 词表内；
3. attention 只能为 0/1，至少包含 CLS、一个内容 token、SEP，且有效位必须是连续前缀；
4. 单句 segment IDs 全为 0；CLS=101、末尾有效位 SEP=102、内容中无额外 SEP，padding token IDs 全为 0；
5. 以上条件全部满足后才转换为 int64。

## 验证

- 17 组非法数组子例覆盖错误 shape、NaN/Inf、小数 token/attention/segment、非法 attention/segment、特殊 token/padding、词表越界、object dtype 和 uint64 越界；均被拒绝。
- 有效的浮点整数数组被接受，转换后的 token IDs 保持原值。
- 单元测试：2/2 通过，含上述 17 组非法子例。
- 在同一固定词表、同一附件4 20 条 PKL 上重跑 probe：20/20 的 token IDs、attention、segment 重放仍完全一致，564 个内容位置的字符 offset 候选不变。原 `results/` 文件与提交前 Git 版本无差异。

该修改仅加固 C-1 审计脚本；没有修改冻结模型、输入特征、映射门或已记录的 285 行状态。
