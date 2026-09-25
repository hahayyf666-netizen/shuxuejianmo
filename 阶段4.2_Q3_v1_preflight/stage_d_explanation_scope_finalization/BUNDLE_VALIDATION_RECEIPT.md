# Q3 Stage D 冻结包验证回执

- 冻结源码 commit：`cb66d39d30622f8649720312d0cc5146b7793ed7`。
- 包：`q3_scope_preflight_server_bundle.zip`；SHA-256：`79c07231aa08c7feb539956005e73ee5a2b60770f8dbdb7a6bed2375a34d3a4a`；大小 `97,377` 字节。
- `verify_bundle.py` 对包内22个受清单约束的文件逐一核对，通过；source manifest SHA-256 为 `1f5a112af760104d4d7e1a2e9666bef4e1b6dcb5723ee8de757231c4d450f56d`。
- 解包后的9项单元测试全部通过。解包后在本机 `Finesse` 使用真实附件2 aligned 数据作了一次完整资格脚本**干运行**，返回PASS，`cuda_available=false`，`formal_training_run=false`。该记录仅证明冻结包入口可执行，**不代替DR-X服务器preflight**。
- 本轮未运行B0/B1正式训练，未用test选择模型，未对附件4正式推理。

DR-X上应按 `SERVER_HANDOFF.md` 重新计算包哈希、解包校验并运行资格脚本。服务器回执与本地干运行回执不可混用。
