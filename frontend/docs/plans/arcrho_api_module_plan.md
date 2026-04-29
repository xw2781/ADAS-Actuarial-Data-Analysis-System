ArcRho 自定义 Python 模块（`arcrho`）子项目计划
版本：v0.2
更新时间：2026-02-17

---

1. 子项目目标

在 Scripting Console 中提供可直接导入的业务模块：

`import arcrho`

目标是给业务用户提供稳定、易用、面向对象的 API，减少对底层文件结构、HTTP 路由和内部数据格式的直接依赖。

核心目标：
1. 提供统一入口：`arcrho.application`。
2. 提供项目 -> ReservingClass -> Dataset/Dfm 的对象化访问链路。
3. 支持常见读写操作（先读后写，分阶段落地）。
4. 保持对现有 scripting 能力兼容，不破坏已有 `/scripting/*` 行为。

---

2. 约束与设计原则

必须遵守：
1. 根目录来源必须是 `workspace_paths.json` / `app_server.config`，禁止硬编码 `E:\\ArcRho`。
2. 代码分层保持 `router -> service -> config/schema`，不要把业务逻辑塞进 router。
3. 对外接口默认向后兼容；新增能力优先增量，不直接重构旧行为。
4. Scripting 会话隔离语义保持不变（不同会话变量互不污染）。
5. 错误返回要可读（参数错误/资源不存在/权限限制/运行时错误需可区分）。

建议策略：
1. `arcrho` 优先封装现有 service 层能力，不直接耦合前端页面逻辑。
2. 写操作先做最小闭环，逐步扩展，避免一次性大范围改动。

---

3. 目标 API 草案（MVP）

3.1 顶层入口

```python
import arcrho

arcrho.application.version
arcrho.application.root
arcrho.projects()
arcrho.project("my_project")
```

3.2 对象层级

```text
Application
  └─ Project
      └─ ReservingClass
          ├─ Dataset
          └─ Dfm
```

3.3 对象与方法（第一版建议）

`Application`
1. `version: str`
2. `root: str`
3. `projects() -> list[Project]`
4. `project(name: str) -> Project`

`Project`
1. `name: str`
2. `folder: str`
3. `reserving_class(path: str) -> ReservingClass`
4. `reserving_classes() -> list[ReservingClass]`（可选）

`ReservingClass`
1. `path: str`
2. `dataset(name_or_id: str) -> Dataset`
3. `datasets() -> list[Dataset]`（可选）
4. `dfm(name: str) -> Dfm`
5. `DFM(name: str) -> Dfm`（兼容别名，不建议在新代码中使用）

`Dfm`
1. `name: str`
2. `data.values(row: int, col: int) -> Any`
3. `ratios.values(row: int | None = None, col: int | None = None) -> Any`
4. `ratios.selected(col: int) -> Any`
5. `ratios.set_selected(col: int, avg_formula_name: str) -> None`
6. `ratios.set_user_entry(col: int, value: float) -> None`

---

4. 示例（文档对外展示口径）

```python
import arcrho

print(arcrho.application.version)
print(arcrho.application.root)

project = arcrho.project("my_project")
print(project.name, project.folder)

reserving_class = project.reserving_class("my_path")
dfm = reserving_class.dfm("my_dfm")

value_1 = dfm.data.values(1, 2)
value_2 = dfm.ratios.values(row=3, col=4)
value_3 = dfm.ratios.selected(col=1)

dfm.ratios.set_selected(col=1, avg_formula_name="simple_3")
dfm.ratios.set_user_entry(col=1, value=1.2345)
```

---

5. 实施范围与非范围

5.1 本期范围（In Scope）
1. 在 scripting 执行环境可用 `import arcrho`。
2. 完成 `Application/Project/ReservingClass/Dfm` 最小对象模型。
3. 支持核心读能力（root、projects、对象获取、基础 values/selected）。
4. 支持最小写能力（`set_selected`、`set_user_entry`）。
5. 提供 API 帮助文档入口（可接入 `/scripting/api-help`）。

5.2 非本期范围（Out of Scope）
1. 一次性覆盖全部 Dfm 高级编辑能力。
2. 在本期内提供复杂富输出渲染（图像、HTML、交互控件）。
3. 重构现有 scripting 路由协议或会话机制。

---

6. 分阶段计划与里程碑

Phase 0：需求冻结与技术设计（0.5-1 天）
1. 确认对象命名、方法签名、错误语义。
2. 确认首期只做哪些写操作。
3. 产出：接口清单 + 验收清单。

Phase 1：模块骨架与只读能力（1-2 天）
1. 建立 `arcrho` 包结构与入口。
2. 实现 `application.root/version/projects/project(...)`。
3. 接入 scripting 会话命名空间，保证多窗口会话隔离不变。
4. 产出：可运行只读 demo。

Phase 2：Dfm 读写最小闭环（2-3 天）
1. 实现 `Dfm.data.values`、`ratios.values/selected`。
2. 实现 `ratios.set_selected`、`ratios.set_user_entry`。
3. 完成异常分类（参数/不存在/权限/运行错误）。
4. 产出：读写端到端可验证。

Phase 3：文档、测试与发布（1-2 天）
1. 单元测试 + 集成测试 + 手工回归脚本。
2. 补充 API 帮助文档与示例 notebook。
3. 发布说明（新增 API、已知限制、回滚方式）。

---

7. 任务分解（执行清单）

App Server
1. 新增 `arcrho` 模块实现（建议放在 `app_server/services` 邻近或独立 package）。
2. 通过 service 层封装数据读写，不绕开 config 规则。
3. 补充必要 schema（如新增专用接口时）。

Scripting
1. 在会话初始化中注入 `arcrho` 可导入能力。
2. 保持现有 `/scripting/run`、`/scripting/run-stream`、`/scripting/interrupt` 兼容。
3. 在 `/scripting/api-help` 暴露新模块常用方法说明。

Docs
1. 更新后端/前端相关 MANUAL 文档（若行为或契约受影响）。
2. 追加 `arcrho_api_module` 示例与错误说明。
3. 运行文档索引构建检查。

QA
1. 新建最小回归脚本（导入、读取、写入、异常、并发会话）。
2. 覆盖路径变更场景（修改 `workspace_paths.workspace_root` 后是否生效）。
3. 覆盖权限场景（禁止写目录的错误提示）。

---

8. 验收标准（Definition of Done）

功能验收：
1. `import arcrho` 在 scripting console 成功。
2. `arcrho.application.root` 与 `workspace_paths` 配置一致。
3. `arcrho.projects()` 可返回项目对象列表，且不依赖硬编码路径。
4. `Dfm` 读方法与写方法在目标样例项目上可用。

兼容性验收：
1. 现有 scripting notebook 执行、保存、加载行为无回归。
2. 现有 `/scripting/*` 路由契约不破坏。
3. 现有 session 隔离语义不变。

质量验收：
1. 关键 API 有单元测试与失败场景测试。
2. 错误信息可读，可定位到对象/字段/参数。
3. 文档示例可直接运行。

---

9. 风险与应对

风险 1：路径逻辑被写死，导致环境迁移失败。
应对：统一通过 `app_server.config` 取根路径，增加路径变更回归用例。

风险 2：对象 API 与现有 service 语义不一致。
应对：先定义 service 适配层，避免在对象层直接拼接文件读写。

风险 3：写操作破坏数据一致性。
应对：写操作先做最小范围，必要时使用临时文件 + 原子替换策略。

风险 4：跨会话污染（变量/状态串台）。
应对：严格复用现有 session_id 隔离机制，不使用全局可变单例缓存业务对象。

---

10. 回滚策略

1. 保留原有 scripting helper API，不移除旧入口。
2. `arcrho` 新功能可通过配置开关禁用（建议增加开关）。
3. 若上线后异常，优先回退 `arcrho` 注入步骤，不影响基础 scripting 执行路径。

---

11. 待确认问题（实施前需要拍板）

1. `Dfm` 的权威数据源
- 决议：以持久化数据文件为权威来源，缓存仅用于加速读取。
- 行为：读取时若不存在目标 `Dfm`，返回明确“未找到”错误，不返回 `None`。
- 创建：支持 `add_dfm()` 创建内存对象，调用 `save()` 后持久化落盘。

2. `set_selected`、`set_user_entry` 的持久化语义
- 决议：两者仅修改内存对象并标记 dirty，不立即落盘。
- 持久化：通过显式 `save()` 原子写入本地 JSON（含锁与并发保护）。

3. 首期 `Dataset` API 范围
- 决议：首期包含最小只读 `Dataset` API（列表、基础元信息、按坐标取值）。
- 说明：`Dataset` 写接口不纳入首期，优先保证 `Dfm` 闭环可用。
