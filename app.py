import streamlit as st
from datetime import datetime, timedelta, time as dtime

import tail_end_strategy as core


def parse_avoid_dates(text):
    if not text:
        return set()
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    return set(parts)


st.set_page_config(page_title="自动尾盘选股", layout="wide")

st.title("自动尾盘选股（本机版）")
st.warning(
    "本程序仅为量化筛选辅助工具，不构成任何投资建议。股市有风险，入市需谨慎。尾盘策略波动剧烈，请严格遵守止损纪律。"
)

def _parse_hhmm(hhmm):
    h, m = hhmm.split(":")
    return dtime(int(h), int(m))


def _next_weekday(date_):
    d = date_
    while d.weekday() >= 5:
        d = d + timedelta(days=1)
    return d


def next_buy_time(now, avoid_dates, is_holiday):
    if is_holiday:
        return None
    start_t = _parse_hhmm(core.TRADING_START)
    end_t = _parse_hhmm(core.TRADING_END)
    d = now.date()
    if now.weekday() < 5 and now.weekday() != 4 and now.strftime("%Y%m%d") not in avoid_dates:
        if now.time() < start_t:
            return datetime.combine(d, start_t)
        if start_t <= now.time() <= end_t:
            return now
    d = d + timedelta(days=1)
    while True:
        d = _next_weekday(d)
        ymd = d.strftime("%Y%m%d")
        if d.weekday() != 4 and ymd not in avoid_dates:
            return datetime.combine(d, start_t)
        d = d + timedelta(days=1)


def next_sell_time(now, is_holiday):
    if is_holiday:
        return None
    start_t = _parse_hhmm(core.SELL_START)
    end_t = _parse_hhmm(core.SELL_END)
    d = now.date()
    if now.weekday() < 5:
        if now.time() < start_t:
            return datetime.combine(d, start_t)
        if start_t <= now.time() <= end_t:
            return now
    d = d + timedelta(days=1)
    d = _next_weekday(d)
    return datetime.combine(d, start_t)


def format_duration(delta):
    total = int(max(0, delta.total_seconds()))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}天")
    if hours or days:
        parts.append(f"{hours}小时")
    if minutes or hours or days:
        parts.append(f"{minutes}分钟")
    parts.append(f"{seconds}秒")
    return "".join(parts)


with st.sidebar:
    st.subheader("全局参数")
    force_run = st.toggle("测试模式（忽略交易时段/周五/避开日）", value=False)
    is_holiday = st.toggle("重大节日/纪念日（今天不交易）", value=False, disabled=force_run)
    avoid_dates_text = st.text_input("额外避开日期（YYYYMMDD，逗号分隔）", value="")
    index_change_pct = st.text_input("上证指数涨跌幅(%)（可留空自动获取）", value="")
    save_to_disk = st.toggle("保存CSV到本地目录", value=True)
    st.divider()
    st.subheader("买入参数")
    target_count = st.slider("目标持仓数量", min_value=1, max_value=6, value=4)
    risk_amount = st.number_input("单票最大亏损金额（用于仓位建议）", min_value=0, value=2000, step=100)
    st.divider()
    st.subheader("卖出参数")
    take_profit_pct = st.number_input("目标收益(%)（用于次日持有判断）", min_value=0.0, value=2.0, step=0.5)
    gap_down_sell_pct = st.number_input("低开阈值(%)（低于则倾向卖出）", value=-1.0, step=0.5)
    st.divider()
    st.subheader("缓存")
    st.write(f"历史缓存目录：`{core.HIST_CACHE_DIR}`")
    clear_today = st.button("清空今日历史缓存")
    clear_all = st.button("清空全部历史缓存")

core.CUSTOM_AVOID_DATES = parse_avoid_dates(avoid_dates_text)

manual_index = None
if index_change_pct.strip():
    try:
        manual_index = float(index_change_pct.strip())
    except Exception:
        st.sidebar.error("上证指数涨跌幅输入无效，请输入数字，例如 -0.8")

now = datetime.now()
current_time = now.strftime("%H:%M")
buy_window = f"{core.TRADING_START}-{core.TRADING_END}"
sell_window = f"{core.SELL_START}-{core.SELL_END}"

if clear_today:
    removed = core.clear_hist_cache(now=now, all_files=False)
    st.sidebar.success(f"已清空今日历史缓存：{removed} 个文件")
if clear_all:
    removed = core.clear_hist_cache(now=now, all_files=True)
    st.sidebar.success(f"已清空全部历史缓存：{removed} 个文件")

buy_allowed = (core.TRADING_START <= current_time <= core.TRADING_END) and (now.weekday() != 4) and (now.strftime("%Y%m%d") not in core.CUSTOM_AVOID_DATES)
sell_allowed = (core.SELL_START <= current_time <= core.SELL_END)
nb = next_buy_time(now, core.CUSTOM_AVOID_DATES, is_holiday=is_holiday)
ns = next_sell_time(now, is_holiday=is_holiday)

with st.container(border=True):
    st.subheader("操作时间")
    c1, c2, c3 = st.columns(3)
    c1.metric("当前时间", current_time)
    c2.metric("尾盘买入窗口", buy_window)
    c3.metric("次日卖出检查窗口", sell_window)

with st.container(border=True):
    st.subheader("权限与倒计时")
    if force_run:
        st.info("测试模式已开启：允许在任意时间运行（仅用于调试）。")
    left, right = st.columns(2)
    with left:
        st.metric("买入操作权限", "允许" if buy_allowed and not is_holiday else "不允许")
        if nb is None:
            st.metric("下一次允许买入", "—")
            st.metric("倒计时", "—")
        else:
            st.metric("下一次允许买入", nb.strftime("%Y-%m-%d %H:%M"))
            st.metric("倒计时", format_duration(nb - now))
    with right:
        st.metric("卖出检查权限", "允许" if sell_allowed and not is_holiday else "不允许")
        if ns is None:
            st.metric("下一次允许卖出检查", "—")
            st.metric("倒计时", "—")
        else:
            st.metric("下一次允许卖出检查", ns.strftime("%Y-%m-%d %H:%M"))
            st.metric("倒计时", format_duration(ns - now))

tab_buy, tab_sell = st.tabs(["尾盘买入选股", "次日卖出检查"])

with tab_buy:
    with st.container(border=True):
        st.subheader("尾盘买入选股")
        if not force_run and (not buy_allowed or is_holiday):
            st.error(f"当前不在允许的尾盘买入时间范围内（{buy_window}），当前 {current_time}。")
        run_buy = st.button("运行尾盘选股", type="primary", disabled=(not force_run and (not buy_allowed or is_holiday)))

    if run_buy:
        with st.spinner("正在运行尾盘选股..."):
            result = core.run_strategy(
                now=now,
                index_change_pct=manual_index,
                is_holiday=is_holiday,
                force_run=force_run,
                target_count=target_count,
                risk_amount=risk_amount,
                save_to_disk=save_to_disk,
                verbose=False,
            )

        if result.get("need_index_input"):
            st.error("无法自动获取大盘数据，请在左侧输入上证指数涨跌幅(%)后再运行。")
            st.caption(result.get("reason", ""))
        elif not result.get("ok"):
            st.error(result.get("reason", "运行失败"))
        else:
            st.success("运行完成")
            st.session_state["last_portfolio"] = result["portfolio"]
            sources = result.get("data_sources", {})
            index_meta = sources.get("index", {})

            with st.container(border=True):
                st.subheader("大盘数据")
                a, b, c, d = st.columns(4)
                a.metric("上证指数涨跌幅(%)", f"{index_meta.get('change_pct')}")
                b.metric("上证指数最新价", f"{index_meta.get('price')}" if index_meta.get("price") is not None else "—")
                c.metric("数据源标称时间", f"{index_meta.get('official_time')}" if index_meta.get("official_time") else "—")
                d.metric("数据来源", f"{index_meta.get('source', '—')}")
                st.info(result.get("market_message", ""))

            with st.container(border=True):
                st.subheader("数据来源与更新时间")
                st.write(f"指数：{index_meta.get('source', '—')}（标称时间：{index_meta.get('official_time', '—')}）")
                spot_meta = sources.get("spot", {})
                st.write(f"实时行情：{spot_meta.get('source', '—')}（标称时间：{spot_meta.get('official_time', '—')}）")
                hist_meta = sources.get("hist", {})
                st.write(
                    f"历史行情：{hist_meta.get('source', '—')}（范围：{hist_meta.get('date_range', '—')}，缓存目录：{hist_meta.get('cache_dir', '—')}）"
                )
                stats = result.get("stats", {})
                st.caption(
                    f"历史缓存命中：{stats.get('hist_cache_hit', 0)}，未命中：{stats.get('hist_cache_miss', 0)}，成功：{stats.get('hist_fetch_success', 0)}，失败：{stats.get('hist_fetch_fail', 0)}"
                )

            with st.container(border=True):
                st.subheader("选股结果")
                df = result["portfolio"]
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    label=f"下载CSV（{result['filename']}）",
                    data=result["csv_bytes"],
                    file_name=result["filename"],
                    mime="text/csv",
                )

with tab_sell:
    with st.container(border=True):
        st.subheader("次日卖出检查")
        st.write("上传昨日尾盘选股导出的 CSV，用于今早盘面检测与持有建议。")
        use_last = st.toggle("使用上一次选股结果（无需上传）", value=False)
        uploaded = None
        if not use_last:
            uploaded = st.file_uploader("上传昨日CSV", type=["csv"], accept_multiple_files=False)
        if uploaded is None:
            if use_last and "last_portfolio" in st.session_state:
                df_hold = st.session_state["last_portfolio"].copy()
            else:
                st.info("未上传文件，且没有可用的上一次选股结果。")
                df_hold = None
        else:
            df_hold = core.pd.read_csv(uploaded, dtype={"代码": str})

    if df_hold is not None:
        required_cols = {"代码", "名称", "现价", "止损价"}
        if not required_cols.issubset(set(df_hold.columns)):
            st.error(f"CSV 缺少必要列：{sorted(list(required_cols))}")
        else:
            with st.container(border=True):
                st.subheader("运行卖出检查")
                if not force_run and (not sell_allowed or is_holiday):
                    st.error(f"当前不在允许的卖出检查时间范围内（{sell_window}），当前 {current_time}。")
                run_sell = st.button("运行次日卖出检查", type="primary", disabled=(not force_run and (not sell_allowed or is_holiday)))

            if run_sell:
                with st.spinner("正在进行次日卖出检查..."):
                    res = core.sell_check(
                        df_hold,
                        now=now,
                        index_change_pct=manual_index,
                        is_holiday=is_holiday,
                        force_run=force_run,
                        take_profit_pct=take_profit_pct,
                        gap_down_sell_pct=gap_down_sell_pct,
                    )
                if res.get("need_index_input"):
                    st.error("无法自动获取大盘数据，请在左侧输入上证指数涨跌幅(%)后再运行。")
                    st.caption(res.get("reason", ""))
                elif not res.get("ok"):
                    st.error(res.get("reason", "运行失败"))
                else:
                    st.success("卖出检查完成")
                    idx = res.get("index", {})

                    with st.container(border=True):
                        st.subheader("大盘数据")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("上证指数涨跌幅(%)", f"{idx.get('change_pct')}")
                        s2.metric("上证指数最新价", f"{idx.get('price')}" if idx.get("price") is not None else "—")
                        s3.metric("数据源标称时间", f"{idx.get('official_time')}" if idx.get("official_time") else "—")
                        s4.metric("数据来源", f"{idx.get('source', '—')}")
                        st.info(res.get("market_message", ""))

                    with st.container(border=True):
                        st.subheader("建议清单")
                        out_df = res["result"]
                        st.dataframe(out_df, use_container_width=True)
                        csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        fn = now.strftime("%Y%m%d_%H%M%S") + "_sell_check.csv"
                        st.download_button("下载卖出建议CSV", data=csv_bytes, file_name=fn, mime="text/csv")
