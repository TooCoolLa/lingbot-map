# Rerun 开发注意事项

> 项目可视化开发的实践经验总结

## 1. 核心 API 调用顺序

### ⚠️ 必须先 `rr.init()` 再 `rr.serve_grpc()`

```python
# ❌ 错误：serve_grpc 会被忽略
rr.serve_grpc(grpc_port=9876)  # WARNING: Rerun is disabled
rr.init("my_app")

# ✅ 正确：先初始化
rr.init("my_app")
grpc_uri = rr.serve_grpc(grpc_port=9876)  # 返回: rerun+http://127.0.0.1:9876/proxy
```

### 完整的无头服务器启动流程

```python
import rerun as rr
import time

# 1. 初始化
rr.init("MyApp")

# 2. 启动 gRPC 服务器
grpc_uri = rr.serve_grpc(grpc_port=9876)

# 3. 启动 Web Viewer 并连接到 gRPC
rr.serve_web_viewer(
    connect_to=grpc_uri,
    web_port=9877,
    open_browser=False,
)

# 4. 发送蓝图
rr.send_blueprint(rrb.Vertical(
    rrb.Spatial3DView(name="3D", origin="world/points"),
))

# 5. 记录数据
for i in range(100):
    rr.set_time("frame", sequence=i)
    rr.log("world/points", rr.Points3D(positions=...))

# 6. 保持进程运行
while True:
    time.sleep(1)
```

## 2. 远程访问配置

### ⚠️ Web Viewer 内部 WebSocket 指向 127.0.0.1

即使 Web Viewer 绑定 `0.0.0.0`，页面内部的 WebSocket 连接仍然使用 `127.0.0.1`。

### SSH 端口映射

需要同时映射 **两个端口**：

```bash
# gRPC 端口 (9876) + Web 端口 (9877)
ssh -L 9877:localhost:9877 -L 9876:localhost:9876 user@server
```

### 正确的访问地址

```
http://localhost:9877/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy
```

| 参数 | 说明 |
|------|------|
| `9877` | Web Viewer 端口 (HTTP 页面) |
| `9876` | gRPC 服务器端口 (数据流) |
| `/proxy` | gRPC 代理路径，必须包含 |

## 3. 三种运行模式对比

| 模式 | 代码 | 适用场景 |
|------|------|----------|
| **本地 Viewer** | `rr.spawn()` | 有 GUI 的桌面环境 |
| **保存文件** | `rr.save("output.rrd")` | 离线查看、分享数据 |
| **在线流式** | `rr.serve_grpc()` + `rr.serve_web_viewer()` | 无头服务器远程查看 |

### CLI 查看 .rrd 文件

```bash
# 本地查看
rerun output.rrd

# Web 模式 (无头服务器)
rerun output.rrd --web-viewer --bind 0.0.0.0 --web-viewer-port 9877
```

## 4. 点云处理注意事项

### 数据格式

```python
# 点云: (N, 3) float32
positions = np.array([[x, y, z], ...])

# 颜色: (N, 3) uint8, 范围 0-255
colors = np.array([[r, g, b], ...])

# 记录
rr.log("world/points", rr.Points3D(
    positions=positions,
    colors=colors,
    radii=0.03,  # 点半径
))
```

### 图像颜色映射

点云的 (h, w) 索引与图像像素**一一对应**，必须先将图像 resize 到点云分辨率：

```python
# 点云分辨率: (H=352, W=640)
# 原始图像: 720x1280
img = np.array(Image.open("image.jpg").resize((640, 352), Image.BILINEAR))
colors = img.reshape(-1, 3)  # (H*W, 3)
```

### 置信度过滤陷阱

**模型输出的置信度可能不在 0-1 范围内！**

```python
# ❌ 如果 99% 的点 conf=1.0，这行无效
mask = conf >= 0.5  # 所有点都通过

# ✅ 使用严格大于
mask = conf > 1.0  # 只保留 conf > 1.0 的点

# ✅ 或限制最大点数 (最实用)
if len(points) > 50000:
    indices = np.random.choice(len(points), 50000, replace=False)
    points = points[indices]
```

### 性能建议

| 场景 | 建议 |
|------|------|
| 单帧点数 | ≤ 50,000 |
| 总帧数 | ≤ 100 帧 |
| 点数过多 | 随机采样 + 设置 `radii` 稍大 |
| 颜色图分辨率 | 压缩 JPEG 质量 `compress(jpeg_quality=85)` |

## 5. 时间轴设置

```python
# 帧序号 (离散时间线)
rr.set_time("frame", sequence=t)

# 真实时间 (连续时间线)
rr.set_time("time", duration=t / fps)  # 秒
```

## 6. 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Rerun is disabled - serve_grpc() call ignored` | 未先调用 `rr.init()` | 确保 `rr.init()` 在 `serve_grpc()` 之前 |
| `Failed to fetch` (Web Viewer) | WebSocket 连错地址 | 使用完整 URL: `http://localhost:PORT/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3APORT%2Fproxy` |
| 页面空白 | 端口未映射或未运行 | 检查 `netstat` 确认端口监听；确认 SSH 映射 |
| 浏览器卡顿 | 点数过多 | 采样到 50K 以下 |
| 颜色错位 | 图像分辨率不匹配 | resize 到点云 (H, W) 再 reshape |

## 7. 常用代码模板

### 最小可运行示例

```python
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import time

rr.init("demo")
grpc_uri = rr.serve_grpc(grpc_port=9876)
rr.serve_web_viewer(connect_to=grpc_uri, web_port=9877, open_browser=False)

rr.send_blueprint(rrb.Spatial3DView(name="3D", origin="world"))

for i in range(100):
    rr.set_time("frame", sequence=i)
    pts = np.random.randn(1000, 3)
    colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
    rr.log("world/points", rr.Points3D(positions=pts, colors=colors, radii=0.05))

while True:
    time.sleep(1)
```

### 后台启动 (screen)

```bash
screen -dmS rerun_vis bash -c 'python visualize.py --headless --serve_port 9876'
# 查看状态
screen -ls
# 查看日志
screen -S rerun_vis -p 0
# 停止
screen -S rerun_vis -X quit
```
