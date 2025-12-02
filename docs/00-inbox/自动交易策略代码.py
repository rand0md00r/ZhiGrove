#coding:gbk
from datetime import datetime
import math
import pymongo
from pymongo import MongoClient
import time
from urllib.parse import quote_plus

# ===================== 1. 核心配置 =====================
# MongoDB配置
DATA_FILTER = {"phase": "竞价", "indicator_name": "旋风一号"}

# 交易配置
TRADE_ACCOUNT = "904800028165"

# 全局状态
IS_TRADED = False              # 当天是否已执行交易
MORNING_SOLD = False           # 是否已执行早盘卖出（9:30）
AFTERNOON_SOLD = False         # 是否已执行尾盘卖出（14:57）
CANCEL_DONE = False            # 是否已执行过 14:55 撤单

AUCTION_BUY_CODES = set()      # 例如 {'001216', '000632', ...}
AUCTION_CANCEL_DONE = False    # 是否已经跑过一次竞价买入撤单

STOP_LOSS_PROCESSED = set()     # 记录已经做过止损处理的股票（存纯代码）

def log_block(title):
    print("\n" + "=" * 40)
    print(f"{title}  时间：{datetime.now().strftime('%H:%M:%S')}")
    print("=" * 40)


# ===================== 2. MongoDB工具函数 =====================
def get_today_stock_basket(max_retries=15, retry_interval=10):
    """从MongoDB读取今日股票篮子（带重试机制）"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 远程MongoDB配置
    remote_username = "zheng123"
    remote_password = "Jack@1357"
    # URL编码用户名和密码（处理特殊字符）
    encoded_username = quote_plus(remote_username)
    encoded_password = quote_plus(remote_password)

    for retry in range(max_retries):
        try:
            client = MongoClient(
                f'mongodb://{encoded_username}:{encoded_password}@118.196.11.76:27017/',
                connectTimeoutMS=30000,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=10000
            )
            client.admin.command("ping")
            
            # 选择数据库和集合
            db = client['stock']
            coll = db['daily_stocks']
            query = {**DATA_FILTER, "date": today}
            
            # 执行查询
            doc = coll.find_one(query, sort=[("update_time", -1)])
            client.close()
            
            # 校验文档有效性并提取股票代码
            if doc and doc.get("date") == today:
                stock_info_list = doc.get("stock_info", [])
                stock_codes = []
                
                if isinstance(stock_info_list, list) and len(stock_info_list) > 0:
                    # 遍历stock_info，提取thscode字段
                    for stock_info in stock_info_list:
                        if isinstance(stock_info, dict) and "thscode" in stock_info:
                            thscode = stock_info.get("thscode")
                            if thscode:  # 过滤空值
                                stock_codes.append(thscode)
                
                # 去重后返回
                return list(set(stock_codes))
            
            time.sleep(retry_interval)
        
        except Exception:
            time.sleep(retry_interval)
    
    # 所有重试失败后返回空列表
    return []

# ===================== 3. 市场与价格工具函数 =====================
def normalize_code(code: str) -> str:
    """统一股票代码比较格式，去掉 .SH/.SZ 等后缀"""
    if not code:
        return ""
    return code.split('.')[0]

def is_prev_day_or_earlier_position(open_date, today_str):
    """
    判断持仓是否为前一交易日及更早建仓（用于次日卖出安全判断）
    - open_date: 持仓上的建仓日期字段（int 或 str），如 20241118 或 "2024-11-18"
    - today_str: 今天日期字符串，格式 "YYYY-MM-DD"
    """
    if open_date is None:
        # 当前账号只有策略仓位，为避免误判，这里默认允许卖出
        # 如果以后账号里有长期仓位，可以改成 False 更保守
        return True
    try:
        today_int = int(today_str.replace('-', ''))
        if isinstance(open_date, int):
            open_int = open_date
        else:
            open_int = int(str(open_date).replace('-', ''))
        # 只要建仓日期早于今天，就认为是“前一日及更早”
        return open_int < today_int
    except Exception:
        # 日期解析失败时，默认允许（同样基于“专用策略账号”的假设）
        return True

def get_current_holdings(ContextInfo):
    """查询当前实际持仓（直接从账户获取，不依赖ContextInfo存储）"""
    holdings = []
    try:
        position_data = get_trade_detail_data(TRADE_ACCOUNT, "stock", "position")
        
        if not isinstance(position_data, list) or len(position_data) == 0:
            print(f" 未查询到任何持仓")
            return holdings
        
        for pos in position_data:
            # 提取持仓核心信息
            pos_stock_code = pos.m_strInstrumentID if hasattr(pos, "m_strInstrumentID") else pos.m_strStockCode
            volume = pos.m_nVolume if hasattr(pos, "m_nVolume") else 0
            if volume < 100:  # 不足1手忽略
                continue
            
            # 转换为完整股票代码（补全市场后缀）
            if pos_stock_code.startswith("60"):
                full_code = f"{pos_stock_code}.SH"
            elif pos_stock_code.startswith(("00", "30")):
                full_code = f"{pos_stock_code}.SZ"
            else:
                full_code = pos_stock_code
            
            # 尝试读取持仓建仓日期
            open_date = None
            if hasattr(pos, "m_nOpenDate"):
                open_date = pos.m_nOpenDate
            elif hasattr(pos, "m_nDate"):
                open_date = pos.m_nDate

            holdings.append({
                "stock_code": full_code,
                "pure_code": normalize_code(pos_stock_code),
                "lots": int(volume // 100),  # 手数
                "shares": volume,            # 股数
                "open_date": open_date       # 建仓日期（用于次日判断）
            })
        
        return holdings
    
    except Exception as e:
        print(f" 查询持仓异常：{str(e)}")
        return holdings

def get_auction_price(ContextInfo, stock_code):
    """获取股票集合竞价末尾价格（9:27-9:30时段有效）"""
    try:
        print(f"\n【竞价价格查询】股票：{stock_code}")
        full_tick = ContextInfo.get_full_tick([stock_code])

        last_price = full_tick[stock_code]["lastPrice"]
        open_price = full_tick[stock_code]['open']
        volume     = full_tick[stock_code]['volume']
        amount     = full_tick[stock_code]['amount']

        print(f"集合竞价数据-{stock_code}：开盘价={open_price}, 最新价={last_price} 成交量={volume}, 成交额={amount}")

        auction_price = open_price

        # 价格校验
        if auction_price <= 0 or math.isnan(auction_price):
            print(f" 竞价价异常：{auction_price}元（可能非竞价时段）")
            return 0.0
        
        print(f" 集合竞价价格：{auction_price:.2f}元")
        return auction_price
    
    except Exception as e:
        print(f" 竞价价查询异常：{str(e)}")
        return 0.0

def get_limit_prices(ContextInfo, stock_code):
    """
    获取股票涨停价、跌停价（适配不同涨跌幅规则）
    - 创业板(30开头)/科创板(68开头)：±20%
    - ST股：±5%
    - 其他主板(60/00开头)：±10%
    """
    try:
        print(f"\n【计算涨跌停价】股票：{stock_code}")
        # 获取前收盘价（lastClose）和最新行情
        full_tick = ContextInfo.get_full_tick([stock_code])
        last_close = full_tick[stock_code]['lastClose']  # 前收盘价
        if last_close <= 0:
            print(f" 前收盘价异常：{last_close}元")
            return 0.0, 0.0
        
        # 1. 提取纯股票代码（去掉.SH/.SZ后缀）
        pure_code = normalize_code(stock_code)
        
        # 2. 判断涨跌幅比例（核心修复逻辑）
        # 先判断是否为ST股（代码含ST，或名称含ST）
        is_st = False
        try:
            # 尝试获取股票名称，判断是否为ST
            stock_name = full_tick[stock_code]['secName']
            if "ST" in stock_name or "*ST" in stock_name:
                is_st = True
        except:
            # 若无法获取名称，仅通过代码兜底（ST股代码无特征，此为备用）
            pass
        
        if is_st:
            # ST股：5%涨跌幅
            up_rate = 1.05
            down_rate = 0.95
        elif pure_code.startswith(("30", "68")):
            # 创业板(30)、科创板(68)：20%涨跌幅
            up_rate = 1.2
            down_rate = 0.8
        else:
            # 主板(60/00)：10%涨跌幅
            up_rate = 1.1
            down_rate = 0.9
        
        # 3. 计算涨跌停价（保留2位小数，符合A股价格规则）
        # 涨停价：向下取整到0.01（避免超过涨停价）
        up_limit = math.floor(last_close * up_rate * 100) / 100
        # 跌停价：向上取整到0.01（避免低于跌停价）
        down_limit = math.ceil(last_close * down_rate * 100) / 100
        
        print(f" 前收盘价={last_close:.2f}元 | 涨跌幅规则={int((up_rate-1)*100)}% | 涨停价={up_limit:.2f}元 | 跌停价={down_limit:.2f}元")
        return up_limit, down_limit
    
    except Exception as e:
        print(f" 计算涨跌停价异常：{str(e)}")
        return 0.0, 0.0

def get_sell1_price(ContextInfo, stock_code):
    full_tick = ContextInfo.get_full_tick([stock_code])
    tick = full_tick[stock_code]

    # askPrice 是多档卖价列表，0 号元素就是卖一价
    ask_prices = tick.get('askPrice', [])
    if not ask_prices:
        print(f"{stock_code} 无卖盘报价（askPrice 为空）")
        return 0.0

    sell1 = ask_prices[0]
    print(f"{stock_code} 卖一价 = {sell1}")
    return sell1

def get_buy1_price(ContextInfo, stock_code):
    full_tick = ContextInfo.get_full_tick([stock_code])
    tick = full_tick[stock_code]
    bid_prices = tick.get("bidPrice", [])
    if isinstance(bid_prices, list) and len(bid_prices) > 0:
        return bid_prices[0]
    return 0.0

def calculate_buy_params(ContextInfo, stock_codes):
    """计算每个股票的买入价格和手数（按资金均分）"""
    buy_params = []
    try:
        # 1. 查询账户可用资金
        print(f"\n【资金查询】账号：{TRADE_ACCOUNT}")
        acct_data = get_trade_detail_data(TRADE_ACCOUNT, "stock", "account")

        if not isinstance(acct_data, list) or len(acct_data) == 0:
            print(f" 未查询到账户资金信息")
            return []
        
        # 提取可用资金（旧版字段：m_dAvailable）
        available_cash = 0.0
        for acct in acct_data:
            if hasattr(acct, "m_dAvailable"):
                available_cash = acct.m_dAvailable
                break
        
        if available_cash <= 100:  # 最低交易门槛
            print(f" 可用资金不足：{available_cash:.2f}元")
            return []
        
        print(f" 可用资金：{available_cash:.2f}元")
        
        # 2. 计算单只股票可用资金（均分+预留手续费）
        stock_count = len(stock_codes)
        if stock_count == 0:
            return []
        
        # 预留千分之3手续费，实际可用资金=总资金×(1-手续费率)/股票数量
        commission_rate = 0.003
        single_stock_cash = available_cash * (1 - commission_rate) / stock_count
        print(f" 单只股票可用资金：{single_stock_cash:.2f}元（含手续费预留）")
        
        # 3. 逐股计算买入价格和手数
        for stock in stock_codes:
            # 获取竞价价格
            auction_price = get_auction_price(ContextInfo, stock)
            if auction_price <= 0:
                continue
            
            # 计算买入价格（竞价价×1.02，向下取整到0.01元）
            buy_price = math.floor(auction_price * 1.02 * 100) / 100
            # 防止买入价低于跌停价（简单校验）
            if buy_price <= auction_price * 0.9:
                print(f" 买入价异常（低于跌停价）：{buy_price:.2f}元，跳过该股票")
                continue
            
            # 计算最大买入股数
            max_shares = single_stock_cash / buy_price
            buy_shares = int(max_shares // 100) * 100
            
            if buy_shares < 100:
                print(f" 资金不足买入1手：{stock}（需{buy_price*100:.2f}元，可用{single_stock_cash:.2f}元）")
                continue
            
            # 记录买入参数
            buy_params.append({
                "stock_code": stock,
                "buy_price": buy_price,
                "lots": buy_shares
            })
            
            print(f" 买入参数：{stock} | 价格{buy_price:.2f}元 | {buy_shares}股")
        
        return buy_params
    
    except Exception as e:
        print(f" 计算买入参数异常：{str(e)}")
        return []

# ===================== 4. 挂单与状态检查函数 =====================
def place_order(ContextInfo, stock_code, buy_price, lots):
    """执行挂单操作（下单类型1102=按价格买入）"""
    try:
        print(f"\n【挂单操作】股票：{stock_code}")
        passorder(23, 1101, TRADE_ACCOUNT, stock_code, 11, buy_price, float(lots), '旋风策略', 2, "自动买入", ContextInfo)
        print(f" 挂单请求发送成功：{stock_code} | 价格{buy_price:.2f}元 | {lots}手")
        return True

    except Exception as e:
        print(f" 挂单失败：{str(e)}")
        return False

def place_sell_order(ContextInfo, stock_code, sell_price, shares):
    """执行卖出挂单（opType=24=股票卖出，返回委托ID）"""
    try:
        print(f"\n【卖出挂单】股票：{stock_code} | 价格={sell_price:.2f}元 | 股数={shares}")
        passorder(24, 1101, TRADE_ACCOUNT, stock_code, 11, sell_price, float(shares), '旋风策略', 2, "自动卖出", ContextInfo)
        
        # 获取卖出委托ID（用于后续撤单）
        sell_order_id = get_latest_order_id_by_api()
        if sell_order_id:
            print(f" 卖出挂单成功 | 委托ID={sell_order_id}")
            return sell_order_id
        else:
            print(f" 卖出挂单成功，但未获取到委托ID")
            return None
    
    except Exception as e:
        print(f" 卖出挂单失败：{str(e)}")
        return None

def cancel_order(ContextInfo, order_id, stock_code):
    """撤销委托（调用QMT cancel函数）"""
    try:
        print(f"\n【撤销委托】股票：{stock_code} | 委托ID={order_id}")
        # 调用QMT cancel函数：参数（委托ID、账号、账号类型、ContextInfo）
        cancel_result = cancel(order_id, TRADE_ACCOUNT, 'stock', ContextInfo)
        if cancel_result:
            print(f" 委托撤销成功")
            return True
        else:
            print(f" 委托撤销失败（可能已成交或不可撤）")
            return False
    
    except Exception as e:
        print(f" 撤单异常：{str(e)}")
        return False

def get_latest_order_id_by_api():
    """用API自带的get_last_order_id获取最新委托ID（无需get_trade_detail_data）"""
    try:
        print(f"【获取最新委托ID】账号：{TRADE_ACCOUNT}")
        # 调用QMT API的get_last_order_id：参数（账号、账号类型、数据类型）
        # 数据类型传'ORDER'表示查委托，返回最新委托ID（无结果时返回'-1'）
        latest_order_id = get_last_order_id(TRADE_ACCOUNT, 'stock', 'order')
        
        if latest_order_id == '-1' or not latest_order_id:
            print(f" 未获取到最新委托ID（返回值：{latest_order_id}）")
            return None
        print(f" 成功获取最新委托ID：{latest_order_id}")
        return latest_order_id
    
    except Exception as e:
        print(f" 获取委托ID异常：{str(e)}")
        return None

def check_order_status(stock_code, expected_price, expected_lots):
    """检查委托状态（基于实际查询到的order_detail字段，仅用API原生函数）"""
    try:
        print(f"\n【委托状态检查】股票：{stock_code}")
        # 1. 用get_last_order_id获取最新委托ID
        order_id = get_latest_order_id_by_api()
        if not order_id:
            print(f" 无有效委托ID，挂单可能未生成")
            return False
        
        # 2. 用get_value_by_order_id查询委托详情（注意数据类型参数传'ORDER'，与API字段匹配）
        order_detail = get_value_by_order_id(order_id, TRADE_ACCOUNT, 'stock', 'ORDER')
        if not order_detail:
            print(f" 委托ID={order_id} 未查询到详情")
            return False
        
        # 3. 校验order_detail是否包含核心字段（从日志提取的关键字段）
        core_fields = [
            "m_strInstrumentID",    # 股票代码
            "m_dLimitPrice",        # 委托价格
            "m_nVolumeTotalOriginal",  # 委托手数
            "m_nOrderStatus"        # 委托状态
        ]
        missing_fields = [f for f in core_fields if not hasattr(order_detail, f)]
        if missing_fields:
            print(f" 委托详情缺少核心字段：{missing_fields}，无法校验")
            return False
        
        # 4. 匹配当前挂单参数（避免混淆历史委托）
        # 股票代码匹配
        expected_code = stock_code.split('.')[0]

        actual_stock = order_detail.m_strInstrumentID
        if actual_stock != expected_code:
            print(f" 委托股票不匹配：实际={actual_stock} | 预期（纯代码）={expected_code} | 原预期={stock_code}")
            return False
        
        # 委托价格匹配（允许0.01元误差，应对浮点精度问题）
        actual_price = order_detail.m_dLimitPrice
        if not math.isclose(actual_price, expected_price, abs_tol=0.01):
            print(f" 委托价格不匹配：实际={actual_price:.2f}元 | 预期={expected_price:.2f}元")
            return False
        
        # 委托手数匹配（整数对比，无精度问题）
        actual_lots = order_detail.m_nVolumeTotalOriginal
        if actual_lots != expected_lots:
            print(f" 委托手数不匹配：实际={actual_lots}手 | 预期={expected_lots}手")
            return False
        
        # 5. 解析委托状态（基于QMT API状态码定义，与日志中EEntrustStatus对应）
        actual_status = order_detail.m_nOrderStatus
        # 有效状态映射（参考API文档：已报/部成/已成视为挂单成功）
        status_map = {
            50: "已报（等待成交）",
            51: "已报待撤",
            55: "部分成交",
            56: "全部成交",
            54: "已撤",
            57: "废单"
        }
        status_desc = status_map.get(actual_status, f"未知状态（状态码：{actual_status}）")
        print(f" 委托校验通过 | 委托ID={order_id} | 状态={status_desc} | 股票={actual_stock} | 价格={actual_price:.2f}元 | 手数={actual_lots}")
        
        # 6. 判断是否为有效委托（已报/部成/已成视为成功）
        return actual_status in (50, 55, 56)
    
    except Exception as e:
        print(f" 委托状态查询异常：{str(e)}")
        return False

def run_morning_sell(ContextInfo, current_date):
    """早盘卖出流程：按开盘价 * 1.091 卖出前一日及更早的持仓"""
    log_block("早盘卖出流程启动")

    holdings = get_current_holdings(ContextInfo)
    if len(holdings) == 0:
        print("【早盘卖出】当前无持仓，结束")
        return

    for hold in holdings:
        stock_code = hold["stock_code"]
        shares = hold["shares"]
        open_date = hold.get("open_date")

        # 只卖出前一日及更早建仓的持仓，避免误卖当日新买入
        if not is_prev_day_or_earlier_position(open_date, current_date):
            print(f"【早盘卖出】跳过 {stock_code}：持仓非前日/更早建仓（open_date={open_date}）")
            continue
        
        # 获取开盘价，按开盘价×1.091挂单（向下取整到0.01）
        try:
            full_tick = ContextInfo.get_full_tick([stock_code])
            zero_price = full_tick[stock_code]['lastClose']  # 今日 0% 基准价（昨收）

            if zero_price <= 0 or math.isnan(zero_price):
                print(f"【早盘卖出】跳过 {stock_code}：0% 基准价异常（{zero_price:.2f}元）")
                continue

            pure_code = normalize_code(stock_code)
            if pure_code.startswith("30"):      # 创业板：涨跌幅 20%
                factor = 1.191
            else:                               # 其它：默认按 10% 档处理
                factor = 1.091

            sell_price = math.floor(zero_price * factor * 100) / 100
            print(f"【早盘卖出】{stock_code} 基准价={zero_price:.2f} * {factor} -> 卖出价={sell_price:.2f}元")

        except Exception as e:
            print(f"【早盘卖出】跳过 {stock_code}：获取基准价异常 - {str(e)}")
            continue
        
        # 执行卖出挂单
        place_sell_order(ContextInfo, stock_code, sell_price, shares)

    print("【早盘卖出】批量挂单结束")

def run_cancel_auction_buys(ContextInfo):
    # TODO 是按查询撤单还是按记录撤单？
    """9:30 后 3 分钟内：撤销未成交的竞价买入委托"""
    log_block("竞价买入撤单流程启动")

    # 今日没有记录任何策略竞价买入标的，直接退出
    if not AUCTION_BUY_CODES:
        print("【竞价撤单】今日无策略竞价买入记录，跳过")
        return False

    order_list = get_trade_detail_data(TRADE_ACCOUNT, "stock", "order")
    if not isinstance(order_list, list) or len(order_list) == 0:
        print("【竞价撤单】未查询到任何委托记录")
        return False

    # 可撤状态：已报 / 已报待撤 / 部分成交
    cancellable_status = {50, 51, 55}
    cancelled_any = False

    for order in order_list:
        # 提取股票代码
        order_code = None
        if hasattr(order, "m_strInstrumentID"):
            order_code = normalize_code(order.m_strInstrumentID)
        elif hasattr(order, "m_strStockCode"):
            order_code = normalize_code(order.m_strStockCode)

        if not order_code:
            continue

        # 只处理“本策略竞价买入过的标的”
        if order_code not in AUCTION_BUY_CODES:
            continue

        # 只处理“买入方向”的委托：OptName 里包含“买”
        opt_name = getattr(order, "m_strOptName", "")
        if not isinstance(opt_name, str):
            opt_name = str(opt_name or "")

        if "买" not in opt_name:
            continue

        status = getattr(order, "m_nOrderStatus", None)
        if status not in cancellable_status:
            # 这里说明要么已完全成交、要么已经撤/废，跳过就行
            continue

        # 获取委托 ID
        order_id = None
        if hasattr(order, "m_strOrderSysID") and order.m_strOrderSysID:
            order_id = order.m_strOrderSysID
        elif hasattr(order, "m_strOrderID") and order.m_strOrderID:
            order_id = order.m_strOrderID

        if not order_id:
            print(f"【竞价撤单】找不到可用委托ID，股票={order_code}，opt={opt_name}，status={status}，跳过")
            continue

        print(f"【竞价撤单】准备撤单 | code={order_code} | opt={opt_name} | status={status} | order_id={order_id}")
        can_cancel = can_cancel_order(order_id, TRADE_ACCOUNT, 'stock')
        print(f"  can_cancel: {can_cancel}")
        if can_cancel:
            cancel_order(ContextInfo, order_id, order_code)
            cancelled_any = True

    if cancelled_any:
        print("【竞价撤单】未成交/部分成交的竞价买入委托已尝试撤单完毕")
    else:
        print("【竞价撤单】没有需要撤的竞价买入委托")

    return cancelled_any

def run_cancel_open_sells(ContextInfo):
    """尾盘前撤销未成交的卖出委托"""
    log_block("14:55 撤销卖出委托流程")

    holdings = get_current_holdings(ContextInfo)
    if len(holdings) == 0:
        print("【撤单】当前无持仓，跳过撤单")
        return

    # 当前持仓对应的纯代码集合，例如：{'600000', '000001', ...}
    holding_codes = {h["pure_code"] for h in holdings}

    order_list = get_trade_detail_data(TRADE_ACCOUNT, "stock", "order")
    if not isinstance(order_list, list) or len(order_list) == 0:
        print("【撤单】无委托记录可撤")
        return

    # 可撤状态：已报 / 已报待撤 / 部分成交
    cancellable_status = {50, 51, 55}

    for order in order_list:
        # 1) 提取股票代码，并与当前持仓匹配
        order_code = None
        if hasattr(order, "m_strInstrumentID"):
            order_code = normalize_code(order.m_strInstrumentID)
        elif hasattr(order, "m_strStockCode"):
            order_code = normalize_code(order.m_strStockCode)

        if not order_code or order_code not in holding_codes:
            continue

        # 2) 用 m_strOptName 判断是否为“卖出”委托
        # 实测值：order.m_strOptName = "限价卖出"
        opt_name = getattr(order, "m_strOptName", "")
        if not isinstance(opt_name, str):
            opt_name = str(opt_name or "")

        # 只处理“卖出”方向的委托：包含“卖”即可（兼容将来可能出现的其他卖出类型）
        if "卖" not in opt_name:
            continue

        # 3) 只撤“可撤状态”的委托
        status = getattr(order, "m_nOrderStatus", None)
        if status not in cancellable_status:
            # 调试时可以打开下面一行，看有哪些状态被跳过
            # print(f"[撤单跳过] code={order_code}, opt={opt_name}, status={status}")
            continue

        # 4) 提取委托 ID（你确认过 m_strOrderSysID 可用）
        order_id = None
        if hasattr(order, "m_strOrderSysID") and order.m_strOrderSysID:
            order_id = order.m_strOrderSysID
        elif hasattr(order, "m_strOrderID") and order.m_strOrderID:
            order_id = order.m_strOrderID

        if not order_id:
            print(f"【撤单】找不到可用委托ID，股票={order_code}，opt={opt_name}，跳过")
            continue

        print(f"【撤单】准备撤单 | code={order_code} | opt={opt_name} | status={status} | order_id={order_id}")
        can_cancel = can_cancel_order(order_id, TRADE_ACCOUNT, 'stock')
        print(f"can_cancel: {can_cancel}")
        cancel_order(ContextInfo, order_id, order_code)

    print("【撤单】尾盘撤单流程完成")

def run_afternoon_sell(ContextInfo):
    """尾盘 14:57 跌停价强行卖出所有持仓"""
    log_block("尾盘卖出流程启动")

    holdings = get_current_holdings(ContextInfo)
    if len(holdings) == 0:
        print("【尾盘卖出】当前无持仓，结束")
        return

    for hold in holdings:
        stock_code = hold["stock_code"]
        shares = hold["shares"]
        
        # 获取跌停价，挂跌停价卖出
        _, down_limit = get_limit_prices(ContextInfo, stock_code)
        if down_limit <= 0:
            print(f"【尾盘卖出】跳过 {stock_code}：跌停价计算异常")
            continue
        
        place_sell_order(ContextInfo, stock_code, down_limit, shares)
        print(f"【尾盘卖出】{stock_code} 挂单 | 跌停价={down_limit:.2f}元 | 股数={shares}")

    print("【尾盘卖出】批量挂单完成")

def run_auction_buy(ContextInfo):
    """竞价买入流程：9:28-9:30 读取 Mongo 股池并按竞价价买入"""
    global AUCTION_BUY_CODES
    
    log_block("竞价买入流程启动")

    # 1. 读取MongoDB股票篮子
    stock_codes = get_today_stock_basket()
    if len(stock_codes) == 0:
        print("【竞价买入】MongoDB 未返回股票篮子，本日不再尝试")
        return False
    
    # 2. 计算买入参数（价格+手数）
    buy_params = calculate_buy_params(ContextInfo, stock_codes)
    if len(buy_params) == 0:
        print("【竞价买入】无有效买入参数，终止交易")
        return False
    
    # 3. 逐股执行挂单+状态检查
    for params in buy_params:
        stock_code = params["stock_code"]
        buy_price = params["buy_price"]
        lots = params["lots"]
        
        print(f"\n[买入流程] {stock_code} | 目标价={buy_price:.2f} | lots={lots}")

        # 挂单
        if not place_order(ContextInfo, stock_code, buy_price, lots):
            continue

        # 记录：这是今天策略用于竞价买入的标的
        AUCTION_BUY_CODES.add(normalize_code(stock_code))

        # 每只股票操作后间隔1秒，避免请求过于频繁
        time.sleep(1)
    
    print("【竞价买入】所有股票买入流程结束")
    return True

# ===================== 5. QMT策略核心函数 =====================
def timer_morning_sell(ContextInfo):
    """每天固定时间触发：早盘卖出"""
    global MORNING_SOLD
    current_date = datetime.now().strftime("%Y-%m-%d")
    print("[TIMER] timer_morning_sell fired")
    run_morning_sell(ContextInfo, current_date)
    MORNING_SOLD = True

def timer_auction_buy(ContextInfo):
    """每天固定时间触发：竞价买入"""
    global IS_TRADED
    print("[TIMER] timer_auction_buy fired")
    if not IS_TRADED:
        run_auction_buy(ContextInfo)
        # 不管成功与否，只要尝试过竞价买入，就视为已执行
        IS_TRADED = True

def timer_cancel_auction_buys(ContextInfo):
    """每天固定时间触发：对竞价买入的未成交委托做统一撤单"""
    global AUCTION_CANCEL_DONE
    print("[TIMER] timer_cancel_auction_buys fired")

    if AUCTION_CANCEL_DONE:
        print("【竞价撤单】已执行过，跳过")
        return

    # 只有今天触发过竞价买入，才有必要撤单
    if IS_TRADED:
        run_cancel_auction_buys(ContextInfo)
    else:
        print("【竞价撤单】今日未执行竞价买入(IS_TRADED=False)，跳过撤单")

    AUCTION_CANCEL_DONE = True

def timer_cancel_open_sells(ContextInfo):
    """每天固定时间触发：14:55 撤销未成交的卖出委托"""
    global CANCEL_DONE
    print("[TIMER] timer_cancel_open_sells fired")

    # 只有早盘卖出触发过（说明有卖单），才有必要跑这一步
    if CANCEL_DONE:
        print("【撤单】已执行过，跳过")
        return

    run_cancel_open_sells(ContextInfo)
    CANCEL_DONE = True

def timer_afternoon_sell(ContextInfo):
    """每天固定时间触发：14:57 尾盘一键卖出所有持仓"""
    global AFTERNOON_SOLD
    print("[TIMER] timer_afternoon_sell fired")

    if AFTERNOON_SOLD:
        print("【尾盘卖出】已执行过，跳过")
        return

    run_afternoon_sell(ContextInfo)
    AFTERNOON_SOLD = True

# === init & handlebar===
def init(ContextInfo):
    """策略初始化（仅执行一次）"""
    global IS_TRADED, MORNING_SOLD, AFTERNOON_SOLD, CANCEL_DONE
    global AUCTION_BUY_CODES, AUCTION_CANCEL_DONE
    global STOP_LOSS_PROCESSED

    IS_TRADED = False
    MORNING_SOLD = False
    AFTERNOON_SOLD = False
    CANCEL_DONE = False
    AUCTION_BUY_CODES = set()
    AUCTION_CANCEL_DONE = False
    STOP_LOSS_PROCESSED = set()

    # 绑定交易账号
    ContextInfo.accID = TRADE_ACCOUNT
    ContextInfo.set_account(TRADE_ACCOUNT)

    print("="*80)
    print(f"【实盘买入策略启动】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置信息：")
    print(f"  - 交易账号：{TRADE_ACCOUNT}")
    print(f"  - 买入规则：竞价价×1.02（向下取整），资金均分")
    print(f"  - 卖出规则：次日9:30开盘价×1.091挂单，14:55未卖撤单，14:57跌停价卖出")
    print("="*80)

    # ---------- 用 run_time 注册定时任务 ----------
    today = datetime.now().strftime("%Y-%m-%d")
    one_day = "1nDay"      # 每 1 天执行一次

    # 1) 早盘卖出
    ContextInfo.run_time("timer_morning_sell", one_day, f"{today} 09:25:00", "SH")

    # 2) 竞价买入
    ContextInfo.run_time("timer_auction_buy", one_day, f"{today} 09:27:30", "SH")

    # 3) 竞价买入撤单
    ContextInfo.run_time("timer_cancel_auction_buys", one_day, f"{today} 09:35:00", "SH")

    # 4) 撤销未成交卖出委托
    ContextInfo.run_time("timer_cancel_open_sells", one_day, f"{today} 14:56:00", "SH")

    # 5) 跌停价强制卖出全部持仓
    ContextInfo.run_time("timer_afternoon_sell", one_day, f"{today} 14:57:00", "SH")


def handlebar(ContextInfo):
    """
    每个bar触发一次：
    - 对当前持仓做止损检查
    - 若价格跌破昨日收盘价的 -2%（即 < 0.98 * lastClose），则：
        1）撤销该股票当前所有“可撤的卖出委托”
        2）按当前卖一价挂出全部持仓
    - 若 >= 0.98 * lastClose，则什么都不做，继续等待涨停卖出
    """
    global STOP_LOSS_PROCESSED

    now_str = datetime.now().strftime("%H:%M:%S")
    # 只在正常交易时段做止损检查（9:30~14:56），避免竞价阶段乱动
    if now_str < "09:30:00" or now_str > "14:56:30":
        return

    # 1. 获取当前持仓
    holdings = get_current_holdings(ContextInfo)
    if len(holdings) == 0:
        # 没有持仓就不用管
        return

    # 2. 一次性把当前所有委托拉出来，构建“卖出可撤单索引”
    order_list = get_trade_detail_data(TRADE_ACCOUNT, "stock", "order")
    if not isinstance(order_list, list):
        order_list = []

    cancellable_status = {50, 51, 55}  # 已报 / 已报待撤 / 部分成交

    # key: 纯代码 '600000'，value: 对应的卖出可撤订单列表
    sell_orders_by_code = {}

    for order in order_list:
        # 提取股票代码（纯代码，不带.SH/.SZ）
        order_code = None
        if hasattr(order, "m_strInstrumentID"):
            order_code = normalize_code(order.m_strInstrumentID)
        elif hasattr(order, "m_strStockCode"):
            order_code = normalize_code(order.m_strStockCode)

        if not order_code:
            continue

        # 只看“卖出”方向委托（OptName 含“卖”）
        opt_name = getattr(order, "m_strOptName", "")
        if not isinstance(opt_name, str):
            opt_name = str(opt_name or "")
        if "卖" not in opt_name:
            continue

        status = getattr(order, "m_nOrderStatus", None)
        if status not in cancellable_status:
            continue

        # 可以撤的卖出单，按股票分组存起来
        sell_orders_by_code.setdefault(order_code, []).append(order)

    # 3. 遍历每一只持仓股票，做止损判断
    for hold in holdings:
        stock_code = hold["stock_code"]  # 带 .SH/.SZ 的代码
        pure_code  = hold["pure_code"]   # 纯代码，比如 '600000'
        shares     = hold["shares"]

        # 已经做过一次止损处理的，不再重复处理（避免频繁撤单+重挂）
        if pure_code in STOP_LOSS_PROCESSED:
            continue

        try:
            full_tick = ContextInfo.get_full_tick([stock_code])
            tick = full_tick[stock_code]

            # 昨日收盘价（0% 基准）
            last_close = tick.get("lastClose", 0.0)
            # 当前最新价
            last_price = tick.get("lastPrice", 0.0)
            # 开盘价仅用于内部参考（不打印）
            # open_price = tick.get("open", 0.0)

            if last_close <= 0 or last_price <= 0:
                # 行情异常，直接跳过，不打印监控日志避免刷屏
                continue

            # 止损线：昨收价的 -2%
            stop_loss_price = last_close * 0.98

            # 价格在止损线之上：继续等待涨停卖出，不做任何操作（不打印）
            if last_price >= stop_loss_price:
                continue

            # ---- 跌破止损线：撤单 + 按卖一价卖出 ----
            print(f"【止损触发】{stock_code} 当前价={last_price:.2f} 低于昨收-2%({stop_loss_price:.2f})，执行撤单+按卖一价卖出")

            # 3.1 撤销该股票所有“可撤的卖出委托”
            if pure_code in sell_orders_by_code:
                for order in sell_orders_by_code[pure_code]:
                    # 提取委托ID
                    order_id = None
                    if hasattr(order, "m_strOrderSysID") and order.m_strOrderSysID:
                        order_id = order.m_strOrderSysID
                    elif hasattr(order, "m_strOrderID") and order.m_strOrderID:
                        order_id = order.m_strOrderID

                    if not order_id:
                        print(f"【止损撤单】{stock_code} 找不到委托ID，跳过某条委托")
                        continue

                    can_cancel = can_cancel_order(order_id, TRADE_ACCOUNT, 'stock')
                    if can_cancel:
                        print(f"【止损撤单】{stock_code} 撤销委托 order_id={order_id}")
                        cancel_order(ContextInfo, order_id, pure_code)
                    else:
                        print(f"【止损撤单】{stock_code} 委托不可撤（order_id={order_id}），可能已成/已撤/废单")

            # 3.2 使用 get_sell1_price 获取当前“卖一价”，按卖一价挂单卖出全部持仓
            sell1 = get_sell1_price(ContextInfo, stock_code)

            # 如果实在拿不到卖一价，就退而求其次，用 lastPrice
            if sell1 is None or sell1 <= 0:
                sell1 = last_price
                print(f"【止损卖出】{stock_code} 未获取到卖一价，退化为按 lastPrice={sell1:.2f} 卖出全部 {shares} 股")
            else:
                print(f"【止损卖出】{stock_code} 按当前卖一价={sell1:.2f} 卖出全部 {shares} 股")

            # 挂卖单（这里 shares 就是股数）
            sell_price = math.floor(sell1 * 100) / 100  # 向下取两位小数
            place_sell_order(ContextInfo, stock_code, sell_price, shares)

            # 标记这只股票已执行过止损，后续不再重复撤单+重挂
            STOP_LOSS_PROCESSED.add(pure_code)

        except Exception as e:
            # 异常还是要打出来，便于排查
            print(f"【止损检查异常】{stock_code}：{str(e)}")
            continue

def stop(ContextInfo):
    """策略停止"""
    print("\n" + "="*80)
    print(f"【策略停止】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"今日交易状态：{'已执行买入操作' if IS_TRADED else '未执行交易'}")
    print("建议：登录QMT交易模块，查看「委托记录」和「持仓」确认最终状态")
    print("="*80)
