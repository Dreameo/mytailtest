# 自动尾盘选股（Streamlit 部署指南）

## 本地运行

1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 启动

```bash
streamlit run app.py
```

默认会在本机打开：`http://localhost:8501`

## Streamlit Cloud 部署（推荐）

1. 把项目放到 GitHub 仓库根目录（至少包含）
- `app.py`
- `tail_end_strategy.py`
- `requirements.txt`
- `runtime.txt`

2. 打开 Streamlit Cloud
- 访问 `https://share.streamlit.io/`（或 Streamlit Cloud 控制台）
- 选择 `New app`
- 连接你的 GitHub 仓库
- `Main file path` 选择：`app.py`
- 点击 `Deploy`

3. 部署后常见注意事项
- Streamlit Cloud 的磁盘是临时的，`.tail_cache` 可能会在重启/休眠后清空；这不会影响功能，只是“当天缓存复用”的加速效果会变弱。
- 该应用依赖公开数据源（新浪/东方财富等），可能会遇到限流或短暂波动；页面已经做了重试与降速。

## 自建服务器部署（可选）

在服务器上执行：

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

然后通过反向代理（Nginx/Caddy）把 8501 暴露为 HTTPS 域名访问。
