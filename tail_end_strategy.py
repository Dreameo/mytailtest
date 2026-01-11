import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import time
import contextlib
import io
from pathlib import Path
import requests
import re
import os
import tempfile

MAX_RISK_PER_TRADE = 0.02
ATR_MULTIPLIER = 2.0
INDEX_DROP_THRESHOLD = -1.0
TRADING_START = "14:20"
TRADING_END = "14:55"
SELL_START = "09:30"
SELL_END = "10:30"
CUSTOM_AVOID_DATES = set()
HIST_RETRY_TIMES = 2
HIST_RETRY_SLEEP_SECONDS = 0.4
HIST_CALL_SILENCE_STDOUT = True
HIST_RATE_LIMIT_SECONDS = 0.03
HIST_CACHE_DIR = ".tail_cache"
HIST_MAX_ADAPTIVE_SLEEP_SECONDS = 0.6

HIST_MEMORY_CACHE = {}
HIST_ADAPTIVE_SLEEP_SECONDS = HIST_RATE_LIMIT_SECONDS


def _default_headers(referer=None):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _requests_get_json(url, params=None, headers=None, timeout=10, retries=2, backoff=0.5):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


def _resolve_hist_cache_dir():
    configured = os.getenv("TAIL_CACHE_DIR") or os.getenv("HIST_CACHE_DIR")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path(HIST_CACHE_DIR),
            Path.home() / ".tail_cache",
            Path(tempfile.gettempdir()) / "tail_cache",
        ]
    )
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("1")
            test_file.unlink()
            return p
        except Exception:
            continue
    p = Path(tempfile.gettempdir()) / "tail_cache"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


HIST_CACHE_DIR = str(_resolve_hist_cache_dir())


def _secid_from_code(stock_code):
    code = normalize_stock_code(stock_code)
    if code is None:
        return None
    if not (code.isdigit() and len(code) == 6):
        return None
    market = "1" if code.startswith("6") else "0"
    return f"{market}.{code}"


def _fetch_spot_em_clist():
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    headers = _default_headers(referer="https://quote.eastmoney.com/")
    params_base = {
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": "f12,f14,f2,f3,f8,f10,f15,f16,f17,f18,f21",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    }
    rows = []
    page_size = 500
    max_pages = 20
    for pn in range(1, max_pages + 1):
        params = {**params_base, "pn": str(pn), "pz": str(page_size)}
        payload = _requests_get_json(url, params=params, headers=headers, timeout=10, retries=2, backoff=0.6)
        data = payload.get("data") if isinstance(payload, dict) else None
        diff = data.get("diff") if isinstance(data, dict) else None
        if not diff:
            break
        rows.extend(diff)
        total = data.get("total")
        if isinstance(total, int) and len(rows) >= total:
            break
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    mapping = {
        "f12": "代码",
        "f14": "名称",
        "f2": "最新价",
        "f3": "涨跌幅",
        "f8": "换手率",
        "f10": "量比",
        "f17": "今开",
        "f18": "昨收",
        "f15": "最高",
        "f16": "最低",
        "f21": "流通市值",
    }
    df = df.rename(columns=mapping)
    for c in ["最新价", "涨跌幅", "换手率", "量比", "今开", "昨收", "最高", "最低", "流通市值"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "名称" in df.columns:
        df["名称"] = df["名称"].fillna("").astype(str)
    if "代码" in df.columns:
        df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df


def _fetch_hist_df_em(stock_code, start_date, end_date):
    secid = _secid_from_code(stock_code)
    if secid is None:
        return None
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    headers = _default_headers(referer="https://quote.eastmoney.com/")
    params = {
        "secid": secid,
        "klt": "101",
        "fqt": "1",
        "beg": "0",
        "end": "20500101",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    payload = _requests_get_json(url, params=params, headers=headers, timeout=10, retries=2, backoff=0.6)
    data = payload.get("data") if isinstance(payload, dict) else None
    klines = data.get("klines") if isinstance(data, dict) else None
    if not klines:
        return None

    start_ymd = str(start_date)
    end_ymd = str(end_date)
    records = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 6:
            continue
        ymd = parts[0].replace("-", "")
        if ymd < start_ymd or ymd > end_ymd:
            continue
        records.append(
            {
                "日期": parts[0],
                "开盘": pd.to_numeric(parts[1], errors="coerce"),
                "收盘": pd.to_numeric(parts[2], errors="coerce"),
                "最高": pd.to_numeric(parts[3], errors="coerce"),
                "最低": pd.to_numeric(parts[4], errors="coerce"),
                "成交量": pd.to_numeric(parts[5], errors="coerce"),
                "成交额": pd.to_numeric(parts[6], errors="coerce") if len(parts) > 6 else None,
                "换手率": pd.to_numeric(parts[10], errors="coerce") if len(parts) > 10 else None,
                "涨跌幅": pd.to_numeric(parts[8], errors="coerce") if len(parts) > 8 else None,
            }
        )
    if not records:
        return None
    return pd.DataFrame(records)

def _log(message, verbose):
    if verbose:
        print(message)


def clear_hist_cache(now=None, all_files=False):
    now = now or datetime.now()
    cache_dir = Path(HIST_CACHE_DIR)
    if not cache_dir.exists():
        return 0
    removed = 0
    day = now.strftime("%Y%m%d")
    for p in cache_dir.glob("hist_*.pkl"):
        if all_files or f"hist_{day}_" in p.name:
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    keys_to_delete = []
    for k in HIST_MEMORY_CACHE.keys():
        if all_files or (isinstance(k, tuple) and len(k) > 0 and k[0] == day):
            keys_to_delete.append(k)
    for k in keys_to_delete:
        HIST_MEMORY_CACHE.pop(k, None)
    return removed


def _parse_sina_hq_var(text):
    m = re.search(r'="([^"]*)"', text)
    if not m:
        return None
    payload = m.group(1)
    parts = payload.split(",")
    return parts


def normalize_stock_code(code):
    if code is None:
        return None
    s = str(code).strip()
    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)
    return s


def fetch_shanghai_index_info(now=None):
    now = now or datetime.now()
    try:
        url = "https://hq.sinajs.cn/list=sh000001"
        headers = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        parts = _parse_sina_hq_var(r.text)
        if not parts or len(parts) < 32:
            raise ValueError("invalid response")
        name = parts[0]
        open_px = float(parts[1]) if parts[1] else None
        prev_close = float(parts[2]) if parts[2] else None
        last_px = float(parts[3]) if parts[3] else None
        high_px = float(parts[4]) if parts[4] else None
        low_px = float(parts[5]) if parts[5] else None
        trade_date = parts[30] if parts[30] else None
        trade_time = parts[31] if parts[31] else None
        official_dt = f"{trade_date} {trade_time}" if trade_date and trade_time else None
        change_pct = None
        if prev_close and last_px:
            change_pct = round((last_px - prev_close) / prev_close * 100, 3)
        return (
            {
                "name": name,
                "change_pct": change_pct,
                "price": last_px,
                "open": open_px,
                "prev_close": prev_close,
                "high": high_px,
                "low": low_px,
                "official_time": official_dt,
                "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Sina hq.sinajs.cn (sh000001)",
            },
            None,
        )
    except Exception as exc:
        try:
            df_index = ak.stock_zh_index_spot_sina()
            sh_index = df_index[df_index["名称"] == "上证指数"]
            if sh_index.empty:
                return None, str(exc)
            change_pct = float(sh_index.iloc[0]["涨跌幅"])
            price = float(sh_index.iloc[0]["最新价"])
            return (
                {
                    "name": "上证指数",
                    "change_pct": change_pct,
                    "price": price,
                    "open": None,
                    "prev_close": None,
                    "high": None,
                    "low": None,
                    "official_time": None,
                    "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "AkShare.stock_zh_index_spot_sina(Sina)",
                },
                None,
            )
        except Exception:
            return None, str(exc)

def evaluate_market_status(index_change_pct, index_price, is_holiday):
    if is_holiday:
        return False, "重大节日/纪念日不交易。"
    if index_change_pct is None:
        return None, "缺少大盘涨跌幅数据，需要手动输入。"
    if index_change_pct < INDEX_DROP_THRESHOLD:
        return (
            False,
            f"大盘风险警告：上证指数跌幅 {index_change_pct}% 超过阈值 {INDEX_DROP_THRESHOLD}% ，建议停止买入。",
        )
    if index_price is None:
        return True, f"上证指数涨跌幅 {index_change_pct:.3f}%（阈值 {INDEX_DROP_THRESHOLD}%），允许交易。"
    return True, f"上证指数最新价 {index_price:.4f}，涨跌幅 {index_change_pct:.3f}%（阈值 {INDEX_DROP_THRESHOLD}%），允许交易。"


def fetch_hist_df(stock_code, start_date, end_date, now=None, stats=None):
    global HIST_ADAPTIVE_SLEEP_SECONDS
    now = now or datetime.now()
    stats = stats if stats is not None else {}

    cache_day = now.strftime("%Y%m%d")
    cache_key = (cache_day, stock_code, start_date, end_date, "qfq")
    if cache_key in HIST_MEMORY_CACHE:
        stats["hist_cache_hit"] = stats.get("hist_cache_hit", 0) + 1
        return HIST_MEMORY_CACHE[cache_key]

    cache_dir = Path(HIST_CACHE_DIR)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    cache_file = cache_dir / f"hist_{cache_day}_{stock_code}_{start_date}_{end_date}_qfq.pkl"
    if cache_file.exists():
        try:
            df = pd.read_pickle(cache_file)
            HIST_MEMORY_CACHE[cache_key] = df
            stats["hist_cache_hit"] = stats.get("hist_cache_hit", 0) + 1
            return df
        except Exception:
            pass

    stats["hist_cache_miss"] = stats.get("hist_cache_miss", 0) + 1
    df_hist = None
    for attempt in range(HIST_RETRY_TIMES + 1):
        try:
            if HIST_CALL_SILENCE_STDOUT:
                with contextlib.redirect_stdout(io.StringIO()):
                    df_hist = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",
                    )
            else:
                df_hist = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )
            if df_hist is not None and not df_hist.empty:
                stats["hist_fetch_success"] = stats.get("hist_fetch_success", 0) + 1
                HIST_ADAPTIVE_SLEEP_SECONDS = max(HIST_RATE_LIMIT_SECONDS, HIST_ADAPTIVE_SLEEP_SECONDS * 0.85)
                break
        except Exception:
            stats["hist_fetch_fail"] = stats.get("hist_fetch_fail", 0) + 1
            HIST_ADAPTIVE_SLEEP_SECONDS = min(HIST_MAX_ADAPTIVE_SLEEP_SECONDS, HIST_ADAPTIVE_SLEEP_SECONDS + 0.08)
        if attempt < HIST_RETRY_TIMES:
            time.sleep(HIST_RETRY_SLEEP_SECONDS * (attempt + 1) + HIST_ADAPTIVE_SLEEP_SECONDS)

    time.sleep(HIST_ADAPTIVE_SLEEP_SECONDS)
    if df_hist is None or df_hist.empty:
        try:
            df_hist = _fetch_hist_df_em(stock_code, start_date=start_date, end_date=end_date)
            if df_hist is not None and not df_hist.empty:
                stats["hist_fetch_success"] = stats.get("hist_fetch_success", 0) + 1
        except Exception:
            stats["hist_fetch_fail"] = stats.get("hist_fetch_fail", 0) + 1
    if df_hist is not None and not df_hist.empty:
        HIST_MEMORY_CACHE[cache_key] = df_hist
        try:
            df_hist.to_pickle(cache_file)
        except Exception:
            pass
    return df_hist


def calculate_technical_indicators(stock_code, start_date, end_date, now=None, stats=None):
    """
    获取个股历史数据并计算技术指标 (MA20, ATR)
    """
    try:
        stock_code = normalize_stock_code(stock_code)
        if stock_code is None:
            return None
        if len(stock_code) != 6 or not stock_code.isdigit():
            return None
        if stock_code[0] not in ("0", "3", "6"):
            return None

        df_hist = fetch_hist_df(stock_code, start_date, end_date, now=now, stats=stats)
        
        if df_hist is None or df_hist.empty or len(df_hist) < 21:
            return None
            
        # 计算 MA20
        df_hist['MA20'] = df_hist['收盘'].rolling(window=20).mean()
        
        # 计算 ATR (14日)
        # TR = Max(High-Low, Abs(High-PreClose), Abs(Low-PreClose))
        df_hist['H-L'] = df_hist['最高'] - df_hist['最低']
        df_hist['H-PC'] = abs(df_hist['最高'] - df_hist['收盘'].shift(1))
        df_hist['L-PC'] = abs(df_hist['最低'] - df_hist['收盘'].shift(1))
        df_hist['TR'] = df_hist[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df_hist['ATR'] = df_hist['TR'].rolling(window=14).mean()
        
        latest = df_hist.iloc[-1]
        
        return {
            "close": latest['收盘'],
            "ma20": latest['MA20'],
            "atr": latest['ATR'],
            "trend_up": latest['收盘'] > latest['MA20']
        }
        
    except Exception:
        # print(f"计算指标失败 {stock_code}: {e}")
        return None

def get_stock_industry(stock_code):
    """
    获取个股所属行业 (用于行业对冲/分散)
    """
    try:
        info = ak.stock_individual_info_em(symbol=stock_code)
        industry_row = info[info['item'] == '行业']
        if not industry_row.empty:
            return industry_row.iloc[0]['value']
    except Exception:
        pass
    return "未知行业"

def get_realtime_data():
    """
    获取A股实时行情数据
    """
    try:
        # 使用 akshare 获取 A 股实时行情
        df = ak.stock_zh_a_spot_em()
        return df, {
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "AkShare.stock_zh_a_spot_em(Eastmoney)",
        }
    except Exception as e:
        ak_err = str(e)
        try:
            df = _fetch_spot_em_clist()
            if df is None or df.empty:
                return pd.DataFrame(), {
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Eastmoney.push2 qt/clist/get",
                    "fallback_from_error": ak_err,
                    "error": "empty response",
                }
            return df, {
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Eastmoney.push2 qt/clist/get",
                "fallback_from_error": ak_err,
            }
        except Exception as ex2:
            return pd.DataFrame(), {
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "AkShare.stock_zh_a_spot_em(Eastmoney) -> Eastmoney.push2 qt/clist/get",
                "error": f"akshare={ak_err}; fallback={ex2}",
            }

def filter_stocks(df):
    """
    根据尾盘交易法核心原则筛选股票
    """
    # 1. 基础过滤：剔除停牌、ST股（名称中包含ST）
    df = df[~df['名称'].str.contains('ST')]
    df = df[~df['名称'].str.contains('退')]
    
    # 2. 涨幅条件：3% < 涨幅 < 5%
    # 尾盘涨幅适中，既有上攻意愿，又避免追高风险
    df = df[(df['涨跌幅'] > 3) & (df['涨跌幅'] < 5)]
    
    # 3. 量能条件：
    # 量比 > 1.2 (当日量能放大)
    # 换手率 > 3% (股性活跃)
    df = df[df['量比'] > 1.2]
    df = df[df['换手率'] > 3]
    
    # 4. 市值条件：流通市值 < 200亿 (中小盘股更具爆发力)
    # 注意：akshare返回的流通市值单位通常为元，需转换
    # 假设 '流通市值' 列存在且单位为元，200亿 = 200 * 10000 * 10000
    if '流通市值' in df.columns:
         df = df[df['流通市值'] < 200 * 100000000]
    
    return df

def check_tail_momentum(stock_code):
    """
    (可选) 检查尾盘30分钟走势
    这里需要获取分时数据，检查14:30后的走势是否向上
    """
    try:
        # 获取当日分时数据
        df_min = ak.stock_zh_a_minute(symbol=stock_code, period='5', adjust='qfq')
        
        # 获取今天的日期字符串
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 筛选今天14:30之后的数据
        # 注意：实际数据格式需根据API返回调整
        tail_data = df_min[df_min['day'].str.contains(today) & (df_min['day'] >= f"{today} 14:30:00")]
        
        if len(tail_data) < 2:
            return False
            
        # 简单判断：收盘价 > 14:30的价格，且呈上升趋势
        start_price = tail_data.iloc[0]['close']
        end_price = tail_data.iloc[-1]['close']
        
        if end_price > start_price * 1.01: # 尾盘拉升超过1%
            return True
            
    except Exception:
        return True # 如果获取失败，暂不剔除，由人工复核
        
    return False

def print_risk_warning():
    """
    打印显性风险提示
    """
    print("\n" + "!" * 60)
    print("【严重风险提示 / RISK WARNING】")
    print("1. 本程序仅为量化筛选辅助工具，不构成任何投资建议。")
    print("2. 股市有风险，入市需谨慎。程序无法预测突发黑天鹅事件。")
    print("3. 尾盘策略波动剧烈，请严格遵守既定止损纪律 (如触发止损线无条件卖出)。")
    print("4. 请务必独立判断大盘环境，切勿在系统性风险下强行交易。")
    print("!" * 60 + "\n")

def sell_check(
    holdings_df,
    now=None,
    index_change_pct=None,
    is_holiday=False,
    force_run=False,
    take_profit_pct=2.0,
    gap_down_sell_pct=-1.0,
):
    now = now or datetime.now()
    current_time = now.strftime("%H:%M")
    if not force_run and not (SELL_START <= current_time <= SELL_END):
        return {
            "ok": False,
            "reason": f"非次日卖出检查时段（{SELL_START}-{SELL_END}），当前 {current_time}，不允许操作。",
        }

    if index_change_pct is None:
        fetched, err = fetch_shanghai_index_info(now=now)
        if fetched is None:
            return {"ok": False, "need_index_input": True, "reason": f"获取大盘数据失败：{err}"}
        index_change_pct = fetched["change_pct"]
        index_info = fetched
    else:
        index_info = {
            "change_pct": index_change_pct,
            "price": None,
            "official_time": None,
            "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "manual",
        }

    is_safe, market_msg = evaluate_market_status(index_change_pct, index_info.get("price"), is_holiday=is_holiday)
    if is_safe is None:
        return {"ok": False, "need_index_input": True, "reason": market_msg}

    stats = {}
    df_spot, spot_meta = get_realtime_data()
    if df_spot.empty:
        return {"ok": False, "reason": "未获取到实时行情数据。"}
    if index_info.get("official_time"):
        spot_meta = {**spot_meta, "official_time": index_info.get("official_time")}

    codes = []
    for x in holdings_df["代码"].tolist():
        c = normalize_stock_code(x)
        if c:
            codes.append(c)
    df_quotes = df_spot[df_spot["代码"].isin(codes)].copy()
    df_quotes = df_quotes.set_index("代码")

    start_date = (now - timedelta(days=100)).strftime("%Y%m%d")
    end_date = now.strftime("%Y%m%d")

    rows = []
    for _, h in holdings_df.iterrows():
        code = normalize_stock_code(h.get("代码"))
        name = h.get("名称")
        buy_price = float(h.get("现价")) if h.get("现价") is not None else None
        stop_loss = float(h.get("止损价")) if h.get("止损价") is not None else None

        q = df_quotes.loc[code] if code in df_quotes.index else None
        cur_price = float(q["最新价"]) if q is not None and "最新价" in q else None
        five_min = float(q["5分钟涨跌"]) if q is not None and "5分钟涨跌" in q else None
        change_pct = float(q["涨跌幅"]) if q is not None and "涨跌幅" in q else None
        open_px = float(q["今开"]) if q is not None and "今开" in q else None
        prev_close = float(q["昨收"]) if q is not None and "昨收" in q else None
        high_px = float(q["最高"]) if q is not None and "最高" in q else None
        low_px = float(q["最低"]) if q is not None and "最低" in q else None

        gap_pct = None
        range_pct = None
        if prev_close and open_px:
            gap_pct = round((open_px - prev_close) / prev_close * 100, 3)
        if prev_close and high_px and low_px:
            range_pct = round((high_px - low_px) / prev_close * 100, 3)

        indicators = calculate_technical_indicators(
            code, start_date=start_date, end_date=end_date, now=now, stats=stats
        )
        ma20 = indicators["ma20"] if indicators else None
        trend_up = indicators["trend_up"] if indicators else None

        pnl_pct = None
        if buy_price and cur_price:
            pnl_pct = round((cur_price - buy_price) / buy_price * 100, 3)

        suggestion = "SELL"
        reason = "默认次日卖出落袋"
        if cur_price is None or buy_price is None:
            suggestion = "CHECK"
            reason = "缺少价格数据，需人工复核"
        else:
            if stop_loss is not None and cur_price <= stop_loss:
                suggestion = "SELL"
                reason = "触发止损"
            else:
                if gap_pct is not None and gap_pct <= float(gap_down_sell_pct) and (pnl_pct is None or pnl_pct < float(take_profit_pct)):
                    suggestion = "SELL"
                    reason = f"低开 {gap_pct}% 低于阈值 {gap_down_sell_pct}%，优先规避风险"
                else:
                    if pnl_pct is not None and pnl_pct >= float(take_profit_pct):
                        if five_min is not None and five_min < 0:
                            suggestion = "SELL"
                            reason = "已达目标收益且短线转弱，锁定利润"
                        elif trend_up is True and (ma20 is None or cur_price >= ma20):
                            suggestion = "HOLD"
                            reason = "趋势仍强且收益达标，可考虑持有"
                        else:
                            suggestion = "SELL"
                            reason = "收益达标但趋势不强，优先落袋"
                    else:
                        if trend_up is True and (ma20 is None or cur_price >= ma20) and (change_pct is None or change_pct >= 0):
                            suggestion = "HOLD"
                            reason = "趋势向上且盘面不弱，可观察继续持有"
                        else:
                            suggestion = "SELL"
                            reason = "趋势/盘面偏弱，按纪律卖出"

        rows.append(
            {
                "代码": code,
                "名称": name,
                "买入价": buy_price,
                "当前价": cur_price,
                "盈亏(%)": pnl_pct,
                "涨幅(%)": change_pct,
                "5分钟涨跌(%)": five_min,
                "低开幅度(%)": gap_pct,
                "振幅(%)": range_pct,
                "今开": open_px,
                "昨收": prev_close,
                "MA20": round(ma20, 3) if ma20 is not None else None,
                "建议": suggestion,
                "理由": reason,
            }
        )

    result_df = pd.DataFrame(rows)
    return {
        "ok": True,
        "now": now,
        "market_message": market_msg,
        "index": index_info,
        "spot": spot_meta,
        "hist": {
            "source": "AkShare.stock_zh_a_hist(Eastmoney)",
            "cache_dir": str(Path(HIST_CACHE_DIR).resolve()),
            "date_range": f"{start_date}-{end_date}",
            "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "stats": stats,
        "result": result_df,
        "windows": {"sell": {"start": SELL_START, "end": SELL_END}},
    }

def run_strategy(
    now=None,
    index_change_pct=None,
    is_holiday=False,
    force_run=False,
    target_count=4,
    risk_amount=2000,
    save_to_disk=True,
    verbose=False,
):
    now = now or datetime.now()
    current_time = now.strftime("%H:%M")
    today_str = now.strftime("%Y%m%d")
    stats = {}
    data_sources = {}

    if not force_run and not (TRADING_START <= current_time <= TRADING_END):
        return {"ok": False, "reason": f"非尾盘买入时段（{TRADING_START}-{TRADING_END}），当前 {current_time}，不允许操作。"}
    if not force_run and now.weekday() == 4:
        return {"ok": False, "reason": "规则限制：周五不交易。"}
    if not force_run and today_str in CUSTOM_AVOID_DATES:
        return {"ok": False, "reason": "规则限制：重大节日/纪念日不交易。"}

    if index_change_pct is None:
        fetched, err = fetch_shanghai_index_info(now=now)
        if fetched is None:
            is_safe, msg = evaluate_market_status(None, None, is_holiday=is_holiday)
            return {
                "ok": False,
                "need_index_input": True,
                "reason": f"获取大盘数据失败：{err}",
                "market_message": msg,
            }
        index_change_pct = fetched["change_pct"]
        data_sources["index"] = fetched

    if "index" not in data_sources:
        data_sources["index"] = {
            "change_pct": index_change_pct,
            "price": None,
            "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "manual",
        }

    is_safe, msg = evaluate_market_status(index_change_pct, data_sources["index"].get("price"), is_holiday=is_holiday)
    if is_safe is None:
        return {"ok": False, "need_index_input": True, "reason": msg}
    if not is_safe:
        return {"ok": False, "reason": msg}

    # 获取全市场数据
    df, spot_meta = get_realtime_data()
    if "index" in data_sources and data_sources["index"].get("official_time"):
        spot_meta = {**spot_meta, "official_time": data_sources["index"]["official_time"]}
    data_sources["spot"] = spot_meta
    
    if df.empty:
        return {"ok": False, "reason": "未获取到数据，请检查网络或API连接。", "data_sources": data_sources}

    # 初步筛选
    candidates = filter_stocks(df)
    
    _log(f"初选符合条件的股票数量: {len(candidates)}", verbose)
    
    if candidates.empty:
        return {"ok": False, "reason": "今日无初选符合条件的股票。"}
        
    enriched_list = []
    
    # 2. 深度分析与指标计算
    _log("正在进行深度分析 (计算技术指标 & 行业归类)...", verbose)
    start_date = (now - timedelta(days=100)).strftime("%Y%m%d")
    end_date = now.strftime("%Y%m%d")
    data_sources["hist"] = {
        "source": "AkShare.stock_zh_a_hist(Eastmoney)",
        "cache_dir": str(Path(HIST_CACHE_DIR).resolve()),
        "date_range": f"{start_date}-{end_date}",
        "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for _, row in candidates.iterrows():
        code = normalize_stock_code(row['代码'])
        name = row['名称']
        
        # 获取技术指标
        indicators = calculate_technical_indicators(code, start_date=start_date, end_date=end_date, now=now, stats=stats)
        if not indicators:
            continue
            
        # 过滤条件：必须在20日均线上方 (趋势向上)
        if not indicators['trend_up']:
            continue
            
        # 获取行业
        industry = get_stock_industry(code)
        
        # 尾盘动量检查 (复用原有逻辑)
        is_momentum_ok = check_tail_momentum(code)
        if not is_momentum_ok:
             continue
             
        # 计算建议仓位 (基于ATR波动率的风控)
        # 假设账户总资金 100,000 (仅用于演示比例)
        # 风险控制：每笔交易亏损不超过总资金的 2%
        # 止损幅度 = 2 * ATR
        # 买入股数 = (总资金 * 2%) / (2 * ATR)
        atr = indicators['atr']
        if pd.isna(atr) or atr == 0:
            position_advice = 0
        else:
            stop_loss_width = ATR_MULTIPLIER * atr
            # 简化计算：建议买入金额 = (10万 * 2%) / (止损幅度/股价) -> 这里的公式其实是 position_size = risk_amount / stop_loss_per_share
            # 假设 risk_amount = 2000
            position_shares = int(risk_amount / stop_loss_width / 100) * 100 # 向下取整到手
            position_advice = position_shares
            
        enriched_list.append({
            "代码": code,
            "名称": name,
            "行业": industry,
            "现价": indicators['close'],
            "涨幅(%)": row['涨跌幅'],
            "换手(%)": row['换手率'],
            "量比": row['量比'],
            "ATR": round(atr, 3),
            "建议仓位(股)": position_advice,
            "止损价": round(indicators['close'] - (ATR_MULTIPLIER * atr), 2)
        })
    
    # 3. 行业分散与优选 (量化对冲思维：构建平衡组合)
    _log("正在进行投资组合优化 (行业分散 & 多股配置)...", verbose)
    df_enriched = pd.DataFrame(enriched_list)
    
    if df_enriched.empty:
        print("经过深度筛选后无符合条件的股票。")
        return {"ok": False, "reason": "经过深度筛选后无符合条件的股票。"}

    # 策略升级：不仅仅选一只，而是构建 3-4 只股票的组合
    # 逻辑：
    # 1. 优先按换手率 (活跃度) 排序
    # 2. 严格执行行业分散，同一行业板块只入选 1 只龙头，防止板块集体杀跌风险 (对冲思维)
    # 3. 目标选出 3-4 只，资金分散配置
    
    df_sorted = df_enriched.sort_values(by="换手(%)", ascending=False)
    
    final_selection = []
    seen_industries = set()
    TARGET_COUNT = int(target_count) # 目标持仓股票数量
    
    for _, row in df_sorted.iterrows():
        if len(final_selection) >= TARGET_COUNT:
            break
            
        industry = row['行业']
        
        # 如果该行业已经有股票入选，则跳过 (除非是未知行业，但尽量避免)
        if industry in seen_industries and industry != "未知行业":
            continue
            
        final_selection.append(row)
        seen_industries.add(industry)
        
    final_portfolio = pd.DataFrame(final_selection)
    
    if final_portfolio.empty:
         print("无法构建有效组合。")
         return {"ok": False, "reason": "无法构建有效组合。"}
    
    # 重新计算建议仓位 (资金均摊)
    # 假设总资金平均分配给选出的股票，但仍需参考单只股票的风险敞口
    # 这里简单处理：如果选出N只，每只分配 总资金/N 的额度，再结合ATR计算手数
    # 为了简化展示，保持原有的基于单笔风险的计算，但提示用户资金分配
    
    cols = ["代码", "名称", "行业", "现价", "涨幅(%)", "换手(%)", "量比", "建议仓位(股)", "止损价"]
    final_portfolio = final_portfolio[cols].copy()
        
    filename = now.strftime("%Y%m%d_%H%M%S") + ".csv"
    csv_bytes = final_portfolio.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    if save_to_disk:
        final_portfolio.to_csv(filename, index=False, encoding='utf-8-sig')
        _log(f"结果已保存至 {filename}", verbose)
    return {
        "ok": True,
        "now": now,
        "index_change_pct": index_change_pct,
        "market_message": msg,
        "filename": filename,
        "portfolio": final_portfolio,
        "csv_bytes": csv_bytes,
        "data_sources": data_sources,
        "stats": stats,
        "windows": {
            "buy": {"start": TRADING_START, "end": TRADING_END},
            "sell": {"start": SELL_START, "end": SELL_END},
        },
    }


def main():
    print_risk_warning()
    now = datetime.now()
    print(f"当前时间: {now.strftime('%H:%M')}")
    result = run_strategy(now=now, save_to_disk=True, verbose=True)
    if result.get("need_index_input"):
        user_in = input("请输入上证指数涨跌幅(%)与是否重大节日(y/n)，以逗号分隔。例如 -0.8,n: ").strip()
        parts = [p.strip() for p in user_in.split(",") if p.strip()]
        change_pct = None
        holiday_flag = "n"
        if parts:
            try:
                change_pct = float(parts[0])
            except Exception:
                change_pct = None
        if len(parts) > 1:
            holiday_flag = parts[1].lower()
        result = run_strategy(
            now=now,
            index_change_pct=change_pct,
            is_holiday=(holiday_flag == "y"),
            save_to_disk=True,
            verbose=True,
        )
    if not result.get("ok"):
        print(result.get("reason", "运行失败"))

if __name__ == "__main__":
    main()
