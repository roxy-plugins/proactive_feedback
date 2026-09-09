# proactive_feedback

Roxy proactive feedback plugin.

## 移动端看板

插件通过 Roxy 的通用移动 UI 生命周期注册“主动反馈”入口，不要求 Agent 核心识别
插件业务。看板聚焦主动消息是否被继续、明确引用和高可信信号；事件展开后按“主动发出、
用户回应、助手继续”显示关联链路。桌面 Dashboard 保留完整审计字段，移动端不复制桌面
表格。

## 跨会话归因

1.2 优先使用 TurnCommitted 的已持久化消息 ID，避免相同文本出现在多个回合时错配。引用回复按 message ID 解析；手机“单独讨论”按 Core 保存的 discussion_source/source_refs 找到原来信，不按相似内容跨会话猜测。仅允许同一用户的 mobile 会话跨会话关联，私密/归档/禁止主动上下文或验证会话不参与。

明确消息引用仍记为 explicit_quote，置信度表示身份关联强度，不表示用户赞同。单独讨论按关联后的原来信与当前回应评分，保留 discussion_source 来源标签。继续保留每条主动消息的首次反馈口径；重放同一回合不删除或重建既有反馈，因此不会重复触发情绪事件。不回填、删除旧反馈，不改变情绪增量算法或长期偏好审核流程。

验证：ROXY_AGENT_ROOT 指向固定 Core 源码，用 pytest 运行 tests；node --test tests/test_mobile_panel.mjs 验证界面合同。运行数据与源码分离，正式安装沿用 Core 的父 turn、attached child、真实只读 oracle 和正常结束后的切换流程。
