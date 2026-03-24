# SleepyTrader

一个 AI 驱动的 ABM（Agent-Based Modeling）股票市场模拟器。

核心目标是用 200 个具备不同交易人格的智能体，结合技术指标与订单流冲击模型，持续生成可视化市场行为，并通过 WebSocket 实时推送到前端大屏。

## 功能亮点

- 200 个交易代理同时参与市场，涵盖趋势、反转、价值、情绪、风险厌恶等多种人格。
- 批处理 LLM 决策：每轮仅对 50 个代理发起一次批量推理调用，降低令牌和速率压力。
- 市场价格引擎：基于 Kyle's lambda 冲击项 + 随机噪声 + 向基本面回归。
- 实时技术指标：MACD、RSI、Bollinger Bands。
- 实时可视化前端：K 线、成交量、MACD、买卖压力、代理决策流、近期成交。
- 多模型供应商回退链路：Groq -> Together -> OpenRouter。

## 技术栈

- 后端：FastAPI, Uvicorn, asyncio, websockets
- 科学计算：NumPy
- HTTP 客户端：aiohttp
- 前端：原生 HTML/CSS/JS + Canvas
- 通信：WebSocket（后端推送 tick、决策状态与交易数据）

## 项目结构

```text
sleepytrader/
├─ backend/
│  ├─ app.py            # FastAPI 入口 + 仿真主循环 + WebSocket 广播
│  ├─ agents.py         # 代理定义、人格配置、组合初始化、决策执行
│  ├─ ai_client.py      # LLM 路由与 JSON 解析（Groq/Together/OpenRouter）
│  ├─ market.py         # 市场状态与价格演化（Kyle 模型）
│  └─ indicators.py     # MACD/RSI/Bollinger 计算
├─ frontend/
│  └─ index.html        # 可视化大屏（Canvas + WebSocket）
├─ requirements.txt
└─ start.sh             # 一键启动脚本（建 venv、装依赖、拉起服务）
```

## 运行方式

### 方式 1：一键启动（推荐）

```bash
chmod +x start.sh
./start.sh
```

脚本会自动完成：

1. 检查 Python 版本
2. 创建 .venv（若不存在）
3. 安装依赖
4. 读取 .env（若存在）
5. 启动 FastAPI 服务

启动后访问：

```text
http://localhost:8000
```

### 方式 2：手动启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd backend
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
```

## 环境变量（可选）

在项目根目录创建 .env：

```env
# 三选一即可；可多配，按优先级回退
GROQ_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

### LLM 优先级

后端决策调用按以下顺序尝试：

1. GROQ_API_KEY
2. TOGETHER_API_KEY
3. OPENROUTER_API_KEY

如果都未配置，系统会自动退化为全量 hold（无交易）模式，仿真仍可运行。

## 仿真机制

### 回合流程

每轮大致执行：

1. 从市场价格历史提取技术指标
2. 广播 agents_deciding 事件给前端
3. 从 200 代理中轮转取 50 个组成批次
4. 将市场上下文 + 代理摘要发给 LLM
5. 执行返回的 buy/sell/hold 决策
6. 根据净买卖量推动价格更新
7. 重算指标并广播 tick
8. 休眠到下一轮（默认 35 秒）

### 关键参数

- 代理总数：200
- 每轮激活数：50
- 轮询间隔：35 秒
- 初始价格：100.0

这些参数定义在 backend/app.py 顶部常量区域。

### 价格更新思想

价格变化近似可理解为：

$$
\Delta P \approx \lambda \cdot \text{OrderImbalance} + \epsilon + \kappa(F - P)
$$

- $\lambda \cdot \text{OrderImbalance}$：订单流冲击（Kyle）
- $\epsilon$：随机噪声
- $\kappa(F - P)$：向基本面回归

## WebSocket 协议

连接地址：

```text
ws://localhost:8000/ws
```

### 服务端消息类型

- init：初始快照（价格、K 线、代理、回合信息）
- agents_deciding：当前轮代理正在推理
- tick：本轮结算后的完整市场数据
- ping：保活

### tick 关键字段

- price：当前价格
- change_pct：近似 24h 变化率
- total_buy / total_sell：本轮成交方向规模
- volume：本轮总量
- candles：K 线数组
- indicators：MACD/RSI/BB 数据
- decisions：本轮代理决策
- orderbook：模拟盘口
- trades：近期成交

## 指标说明

- MACD：12/26 EMA 差值 + 9 EMA 信号线 + 柱状图
- RSI：14 周期相对强弱
- Bollinger：20 周期均线与 2 倍标准差带

前端会根据这些数据实时绘制图形并更新头部状态。

## 性能与稳定性建议

- 若免费模型限速明显，可增加回合间隔，或进一步降低每轮激活代理数。
- 若网络环境对 TLS 校验严格，注意 aiohttp 请求链路与代理设置。
- 建议将前端和后端放在同机或低延迟网络，避免 WebSocket 抖动影响体验。

## 常见问题

### 1) 页面打不开 / 连接失败

- 确认服务已启动并监听 8000。
- 检查本机防火墙或端口占用。

### 2) 决策一直是 hold

- 检查是否正确配置任一 API Key。
- 检查网络是否可访问对应模型供应商。

### 3) 前端显示但图表不动

- 查看后端日志是否持续输出 round 信息。
- 检查浏览器控制台是否有 WebSocket 异常。

## 二次开发指南

### 扩展代理人格

编辑 backend/agents.py 中 PERSONALITIES 列表，即可注入新的行为风格。

### 调整市场微结构

编辑 backend/market.py：

- 价格冲击系数
- 噪声强度
- 基本面回归强度
- 基本面漂移逻辑

### 调整 LLM 策略

编辑 backend/ai_client.py：

- SYSTEM_PROMPT
- 模型名称
- temperature / max_tokens
- JSON 解析与容错规则

### 调整可视化

编辑 frontend/index.html：

- 图表布局
- 决策流样式
- 动画与主题
- 悬浮提示与交互

## 生产化改造建议

- 增加配置层：将回合参数、模型参数、代理规模放入配置文件。
- 增加日志与观测：结构化日志、请求耗时、模型错误率、tick 处理耗时。
- 增加测试：指标计算单测、订单执行单测、WS 协议回归测试。
- 增加持久化：把价格序列、决策、交易记录落地数据库。
- 增加安全控制：API Key 管理、限流、中间件鉴权。

## License

仅用于学习与研究示例。
