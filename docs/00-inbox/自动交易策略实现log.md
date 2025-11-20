``` python 
#coding:gbk
from datetime import datetime
import math
import pymongo
from pymongo import MongoClient
import time

# ===================== 1. 核心配置 =====================
# MongoDB配置
MONGO_URI = "mongodb://dyx:qx_dyx@192.168.1.142:27017/?authSource=stock"
DB_NAME = "stock"
COLLECTION_NAME = "ths_realtime_stocks"
# DATA_FILTER = {"phase": "竞价", "indicator_name": "千竞一号"}
DATA_FILTER = {"phase": "竞价", "indicator_name": "千尾一号"}

# 交易配置
TRADE_ACCOUNT = "904800028165"
PRICE_MULTIPLE = 1.02          # 竞价价×1.02作为买入价
SYNC_DELAY = 3                 # 数据同步等待时间（秒）

# 全局状态
IS_TRADED = False              # 当天是否已执行交易
MORNING_SOLD = False           # 是否已执行早盘卖出（9:30）
AFTERNOON_SOLD = False         # 是否已执行尾盘卖出（14:57）
CANCEL_DONE = False            # 是否已执行过 14:55 撤单

# 调试开关（方便在 QMT 里手动触发单个流程）
DEBUG_FORCE_MORNING_SELL   = False   # True 时忽略时间条件，强制执行早盘卖出
DEBUG_FORCE_CANCEL_SELL    = False   # True 时忽略时间条件，强制执行 14:55 撤单
DEBUG_FORCE_AFTERNOON_SELL = False   # True 时忽略时间条件，强制执行尾盘卖出
DEBUG_FORCE_AUCTION_BUY    = False   # True 时忽略时间条件，强制执行竞价买入

AUCTION_BUY_CODES = set()      # 例如 {'001216', '000632', ...}
AUCTION_CANCEL_DONE = False    # 是否已经跑过一次竞价买入撤单

def log_block(title):
    print("\n" + "=" * 40)
    print(f"{title}  时间：{datetime.now().strftime('%H:%M:%S')}")
    print("=" * 40)


# ===================== 2. MongoDB工具函数 =====================
def get_today_stock_basket(max_retries=18, retry_interval=10):
    """从MongoDB读取今日股票篮子（带重试机制）"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    for retry in range(max_retries):
        try:
            print(f"\n【MongoDB查询】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（重试{retry+1}/{max_retries}）")
            client = MongoClient(
                MONGO_URI,
                connectTimeoutMS=30000,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=10000
            )
            client.admin.command("ping")
            
            db = client[DB_NAME]
            coll = db[COLLECTION_NAME]
            query = {**DATA_FILTER, "date": today}
            
            doc = coll.find_one(query, sort=[("update_time", -1)])
            client.close()
            
            # 校验文档有效性
            if doc and doc.get("date") == today:
                stock_codes = doc.get("filter_code_list", [])
                if not isinstance(stock_codes, list):
                    stock_codes = []
                
                stock_codes = list(set(stock_codes))
                
                print(f" 获取今日股票篮子：{stock_codes}（共{len(stock_codes)}只）")
                return stock_codes
            
            else:
                print(f"警告：未找到今日（{today}）有效股票篮子，{retry_interval}秒后重试")
                time.sleep(retry_interval)
        
        except Exception as e:
            print(f"获取股票篮子失败：{str(e)}，{retry_interval}秒后重试")
            time.sleep(retry_interval)
    
    # 所有重试失败后返回空列表
    print(f"所有重试失败，未找到今日股票篮子")
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
        print(f"\n【查询当前持仓】账号：{TRADE_ACCOUNT}")
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
        
        print(f" 当前持仓：{[h['stock_code'] for h in holdings]}（共{len(holdings)}只）")
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
    """获取股票涨停价、跌停价（按A股10%幅度计算，前收盘价为基准）"""
    try:
        print(f"\n【计算涨跌停价】股票：{stock_code}")
        # 获取前收盘价（lastClose）和最新行情
        full_tick = ContextInfo.get_full_tick([stock_code])
        last_close = full_tick[stock_code]['lastClose']  # 前收盘价
        if last_close <= 0:
            print(f" 前收盘价异常：{last_close}元")
            return 0.0, 0.0
        
        # A股涨停价=前收盘价×1.1（向下取整到0.01），跌停价=前收盘价×0.9（向上取整到0.01）
        up_limit = math.floor(last_close * 1.1 * 100) / 100  # 涨停价
        down_limit = math.ceil(last_close * 0.9 * 100) / 100  # 跌停价（避免低于跌停价）
        
        print(f" 前收盘价={last_close:.2f}元 | 涨停价={up_limit:.2f}元 | 跌停价={down_limit:.2f}元")
        return up_limit, down_limit
    
    except Exception as e:
        print(f" 计算涨跌停价异常：{str(e)}")
        return 0.0, 0.0

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
            buy_price = math.floor(auction_price * PRICE_MULTIPLE * 100) / 100
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
        # passorder参数（旧版纯位置传参，共10个参数）
        # 参数顺序：opType(23=买入), orderType(1102=按价格), 账号, 股票代码, prType(11=指定价), 
        #          挂单价格, 手数(double), 市场代码(int), 策略名称, 消息, ContextInfo
        passorder(23, 1101, TRADE_ACCOUNT, stock_code, 11, buy_price, float(lots), '千竞策略', 2, "实盘买入", ContextInfo)
        print(f" 挂单请求发送成功：{stock_code} | 价格{buy_price:.2f}元 | {lots}手")
        return True

    except Exception as e:
        print(f" 挂单失败：{str(e)}")
        return False

def place_sell_order(ContextInfo, stock_code, sell_price, shares):
    """执行卖出挂单（opType=24=股票卖出，返回委托ID）"""
    try:
        print(f"\n【卖出挂单】股票：{stock_code} | 价格={sell_price:.2f}元 | 股数={shares}")
        # 调用passorder卖出：opType=24（卖出），orderType=1101（按手数），prType=11（指定价）
        passorder(24, 1101, TRADE_ACCOUNT, stock_code, 11, sell_price, float(shares), '千竞策略', 2, "实盘卖出", ContextInfo)
        
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

# def check_position_status(stock_code, expected_lots):
#     """检查持仓状态（判断是否成功买入，expected_lots 实际为股数）"""
#     try:
#         print(f"\n【持仓状态检查】股票：{stock_code}")
#         position_data = get_trade_detail_data(TRADE_ACCOUNT, "stock", "position")
        
#         if not isinstance(position_data, list) or len(position_data) == 0:
#             print(f" 未查询到持仓记录")
#             return False
        
#         target_position = None
#         # 注意：这里传进来的 expected_lots 实际就是“股数”（已经是 100 的整数倍）
#         expected_shares = expected_lots
#         pure_code = stock_code.split('.')[0]  # 新增：去掉 .SH/.SZ 后缀
#         for pos in position_data:
#             # 优先用API标准字段m_strInstrumentID匹配
#             pos_stock_code = pos.m_strInstrumentID if hasattr(pos, "m_strInstrumentID") else pos.m_strStockCode
#             if (pos_stock_code == pure_code and
#                 hasattr(pos, "m_nVolume") and pos.m_nVolume >= expected_shares):
#                 target_position = pos
#                 break
        
#         if target_position:
#             cost_price = target_position.m_dOpenPrice if hasattr(target_position, "m_dOpenPrice") else 0.0
#             print(f" 买入成功：{stock_code} | 持仓{target_position.m_nVolume}股 | 成本{cost_price:.2f}元")
#             return True

#         else:
#             print(f" 未找到目标持仓")
#             return False
        
    
#     except Exception as e:
#         print(f" 持仓状态查询异常：{str(e)}")
#         return False

#     return False

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

    IS_TRADED = False
    MORNING_SOLD = False
    AFTERNOON_SOLD = False
    CANCEL_DONE = False
    AUCTION_BUY_CODES = set()
    AUCTION_CANCEL_DONE = False

    # 绑定交易账号
    ContextInfo.accID = TRADE_ACCOUNT
    ContextInfo.set_account(TRADE_ACCOUNT)

    print("="*80)
    print(f"【实盘买入策略启动】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置信息：")
    print(f"  - MongoDB：{MONGO_URI.split('@')[-1]}/{DB_NAME}/{COLLECTION_NAME}")
    print(f"  - 交易账号：{TRADE_ACCOUNT}")
    print(f"  - 买入规则：竞价价×{PRICE_MULTIPLE}（向下取整），资金均分")
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
    pass

def stop(ContextInfo):
    """策略停止"""
    print("\n" + "="*80)
    print(f"【策略停止】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"今日交易状态：{'已执行买入操作' if IS_TRADED else '未执行交易'}")
    print("建议：登录QMT交易模块，查看「委托记录」和「持仓」确认最终状态")
    print("="*80)

```

自动交易策略流程图：

1. 读取MongoDB篮子股票信息；
2. 计算每个股票的买单信息：按价格买入（1102），计算买入价格（竞价末尾位置的1.02，向下取整）；
3. 挂单；
4. 检查委托列表，判断是否成功挂单；
5. 检查买入列表，判断是否成功买入；

6. 卖出逻辑：第二天早盘9：30开始挂出卖单，按开盘价格 * 1.091卖出；下午2:55检查是否完成卖出，如果没卖出，撤单，2:57尾盘竞价开始时，挂跌停价卖出；



--- 
# 11月12日 问题

1. 27:00 查询到的数据是错误的个股；
2. buy_price错误，

## 日志

[2025-11-13 09:08:12][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-13 09:08:12][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-13 09:08:12
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 最大买入股票：5只
================================================================================

[2025-11-13 09:28:06][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-13 09:28:06（重试1/18）

[2025-11-13 09:28:06][新建策略文件][SH000300][1分钟]  获取今日股票篮子：['002451.SZ', '000973.SZ']（共2只）

【资金查询】账号：904800028165
? 可用资金：10000.00元
? 单只股票可用资金：4985.00元（含手续费预留）

【竞价价查询】股票：002451.SZ
? 竞价价查询异常：name 'tick_data' is not defined

【竞价价查询】股票：000973.SZ
? 竞价价查询异常：name 'tick_data' is not defined
?? 无有效买入参数，终止交易

[2025-11-13 09:29:13][新建策略文件][SH000300][1分钟] 0D:\国投核心客户极速策略交易终端（ACT）\python\新建策略文件.py_00030013: 策略停止
[2025-11-13 09:29:13][新建策略文件][SH000300][1分钟] 
================================================================================
【策略停止】时间：2025-11-13 09:29:13
今日交易状态：已执行买入操作
建议：登录QMT交易模块，查看「委托记录」和「持仓」确认最终状态
================================================================================

[2025-11-13 09:30:11][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-13 09:30:11][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-13 09:30:11
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 最大买入股票：5只
================================================================================

[2025-11-13 09:30:25][新建策略文件][SH000300][1分钟] 0D:\国投核心客户极速策略交易终端（ACT）\python\新建策略文件.py_00030024: 策略停止
[2025-11-13 09:30:25][新建策略文件][SH000300][1分钟] 
================================================================================
【策略停止】时间：2025-11-13 09:30:25
今日交易状态：未执行交易
建议：登录QMT交易模块，查看「委托记录」和「持仓」确认最终状态
================================================================================

[2025-11-13 09:31:01][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-13 09:31:01][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-13 09:31:01
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 最大买入股票：5只
================================================================================

【MongoDB查询】时间：2025-11-13 09:31:01（重试1/18）

[2025-11-13 09:31:01][新建策略文件][SH000300][1分钟]  获取今日股票篮子：['300497.SZ']（共1只）

【资金查询】账号：904800028165
? 可用资金：10000.00元
? 单只股票可用资金：9970.00元（含手续费预留）

【竞价价查询】股票：300497.SZ
集合竞价数据-300497.SZ：开盘价=17.01, 最新价=17.5 成交量=202298, 成交额=350095000.0
? 集合竞价价格：17.01元
? 买入参数：300497.SZ | 价格17.35元 | 500手（50000股）

【挂单操作】股票：300497.SZ
market_code: 2
float(lots): 500.0
?? 挂单请求发送成功：300497.SZ | 价格17.35元 | 500手

[2025-11-13 09:31:04][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：300497.SZ
?? 未找到目标委托，当前所有委托：
? 委托状态查询异常：'COrderDetail' object has no attribute 'm_strOrderCode'

[2025-11-13 09:31:05][新建策略文件][SH000300][1分钟] 
? 所有股票买入操作已完成


##################################################################################################################

--- 

# 11月13日

## 问题：
1. 买入：千尾；
2. 卖出：千竞 = 涨停价格 * 0.97，向下取整； 千尾 = 涨停价格 * 0.91，向上取整； 没涨停，尾盘直接卖出；
3. 买入的有效检查 = 查委托 + 查持仓；
4. 卖出逻辑：第二天早盘9：30开始挂出卖单，按涨停价格 * 0.91卖出；下午2:55检查是否完成卖出，如果没卖出，撤单，2:57尾盘竞价开始时，挂跌停价卖出；

## 日志：

[2025-11-14 09:17:57][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-14 09:17:57][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-14 09:17:57
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 卖出规则：次日9:30涨停价×0.91挂单，14:55未卖撤单，14:57跌停价卖出
================================================================================

[2025-11-14 09:28:04][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-14 09:28:04（重试1/18）

[2025-11-14 09:28:04][新建策略文件][SH000300][1分钟]  获取今日股票篮子：['000572.SZ', '000688.SZ']（共2只）

【资金查询】账号：904800028165
 可用资金：9999.00元
 单只股票可用资金：4984.50元（含手续费预留）

【竞价价格查询】股票：000572.SZ
集合竞价数据-000572.SZ：开盘价=10.5, 最新价=10.5 成交量=146411, 成交额=153731600.0
 集合竞价价格：10.50元
 买入参数：000572.SZ | 价格10.71元 | 400手（40000股）

【竞价价格查询】股票：000688.SZ
集合竞价数据-000688.SZ：开盘价=25.0, 最新价=25.0 成交量=17923, 成交额=44808400.0
 集合竞价价格：25.00元
 买入参数：000688.SZ | 价格25.50元 | 100手（10000股）

【挂单操作】股票：000572.SZ
market_code: 2
float(lots): 400.0
 挂单请求发送成功：000572.SZ | 价格10.71元 | 400手

[2025-11-14 09:28:07][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：000572.SZ
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM853127
 委托校验通过 | 委托ID=MM853127 | 状态=已报（等待成交） | 股票=000572 | 价格=10.71元 | 手数=400

【持仓状态检查】股票：000572.SZ
 未查询到持仓记录

[2025-11-14 09:28:08][新建策略文件][SH000300][1分钟] 
【挂单操作】股票：000688.SZ
market_code: 2
float(lots): 100.0
 挂单请求发送成功：000688.SZ | 价格25.50元 | 100手

[2025-11-14 09:28:11][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：000688.SZ
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM853131
 委托校验通过 | 委托ID=MM853131 | 状态=已报（等待成交） | 股票=000688 | 价格=25.50元 | 手数=100

【持仓状态检查】股票：000688.SZ
 未查询到持仓记录

[2025-11-14 09:28:12][新建策略文件][SH000300][1分钟] 
 所有股票买入操作已完成



---

# 11月17日 

## 日志：

[2025-11-17 09:11:03][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-17 09:11:03][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-17 09:11:03
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 卖出规则：次日9:30涨停价×0.91挂单，14:55未卖撤单，14:57跌停价卖出
================================================================================

[2025-11-17 09:28:00][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-17 09:28:00（重试1/18）

[2025-11-17 09:28:00][新建策略文件][SH000300][1分钟] 警告：未找到今日（2025-11-17）有效股票篮子，10秒后重试

[2025-11-17 09:28:10][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-17 09:28:10（重试2/18）

[2025-11-17 09:28:10][新建策略文件][SH000300][1分钟] 警告：未找到今日（2025-11-17）有效股票篮子，10秒后重试

[2025-11-17 09:28:20][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-17 09:28:20（重试3/18）

[2025-11-17 09:28:20][新建策略文件][SH000300][1分钟] 警告：未找到今日（2025-11-17）有效股票篮子，10秒后重试

[2025-11-17 09:28:30][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-17 09:28:30（重试4/18）

[2025-11-17 09:28:30][新建策略文件][SH000300][1分钟]  获取今日股票篮子：['603931.SH']（共1只）

【资金查询】账号：904800028165
 可用资金：13209.35元
 单只股票可用资金：13169.72元（含手续费预留）

【竞价价格查询】股票：603931.SH
集合竞价数据-603931.SH：开盘价=32.0, 最新价=32.0 成交量=4901, 成交额=15683200.0
 集合竞价价格：32.00元
 买入参数：603931.SH | 价格32.64元 | 400手（40000股）

【挂单操作】股票：603931.SH
market_code: 1
float(lots): 400.0
 挂单请求发送成功：603931.SH | 价格32.64元 | 400手

[2025-11-17 09:28:33][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：603931.SH
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM567559
 委托校验通过 | 委托ID=MM567559 | 状态=已报（等待成交） | 股票=603931 | 价格=32.64元 | 手数=400

【持仓状态检查】股票：603931.SH
 未找到目标持仓

[2025-11-17 09:28:34][新建策略文件][SH000300][1分钟] 
 所有股票买入操作已完成


## 问题：
1. 分成两个账号轮流交易，每次记录两个分仓的信息；
2. 卖出价格是开盘价的1.091，不是涨停价的0.91
3. 持仓信息不能用代码中的函数记录，因为每天代码会被关闭，只有早上快开盘的时候才开始运行，所以持仓信息需要直接查询
4. mongoDB股票信息刷新错误，最后一次更新在update_time 2025-11-17T09:32:32.473+00:00



---

# 11月18日
 
## 日志：

[2025-11-18 09:21:23][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-18 09:21:23][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-18 09:21:23
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 卖出规则：次日9:30开盘价×1.091挂单，14:55未卖撤单，14:57跌停价卖出
================================================================================

[2025-11-18 09:28:01][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-18 09:28:01（重试1/18）

[2025-11-18 09:28:01][新建策略文件][SH000300][1分钟]  获取今日股票篮子：['600829.SH', '000636.SZ']（共2只）

【资金查询】账号：904800028165
 可用资金：7417.83元
 单只股票可用资金：3697.79元（含手续费预留）

【竞价价格查询】股票：600829.SH
集合竞价数据-600829.SH：开盘价=16.7, 最新价=16.7 成交量=51102, 成交额=85340500.0
 集合竞价价格：16.70元
 买入参数：600829.SH | 价格17.03元 | 200手（20000股）

【竞价价格查询】股票：000636.SZ
集合竞价数据-000636.SZ：开盘价=17.830000000000002, 最新价=17.830000000000002 成交量=24239, 成交额=43218100.0
 集合竞价价格：17.83元
 买入参数：000636.SZ | 价格18.18元 | 200手（20000股）

【挂单操作】股票：600829.SH
market_code: 1
float(lots): 200.0
 挂单请求发送成功：600829.SH | 价格17.03元 | 200手

[2025-11-18 09:28:04][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：600829.SH
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM575061
 委托校验通过 | 委托ID=MM575061 | 状态=已报（等待成交） | 股票=600829 | 价格=17.03元 | 手数=200

【持仓状态检查】股票：600829.SH
 未找到目标持仓

[2025-11-18 09:28:05][新建策略文件][SH000300][1分钟] 
【挂单操作】股票：000636.SZ
market_code: 2
float(lots): 200.0
 挂单请求发送成功：000636.SZ | 价格18.18元 | 200手

[2025-11-18 09:28:08][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：000636.SZ
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM872166
 委托校验通过 | 委托ID=MM872166 | 状态=已报（等待成交） | 股票=000636 | 价格=18.18元 | 手数=200

【持仓状态检查】股票：000636.SZ
 未找到目标持仓

[2025-11-18 09:28:09][新建策略文件][SH000300][1分钟] 
 所有股票买入操作已完成

[2025-11-18 09:30:00][新建策略文件][SH000300][1分钟] 
【早盘卖出】时间窗口触发（09:30:00）

【查询当前持仓】账号：904800028165
 当前持仓：['603931.SH']（共1只）

【卖出挂单】股票：603931.SH | 价格=33.16元 | 手数=4
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM872166
 卖出挂单成功 | 委托ID=MM872166
【早盘卖出】批量挂单完成，等待成交

[2025-11-18 09:31:00][新建策略文件][SH000300][1分钟] 
【早盘成交监控】时间：09:31:00（每30秒检查一次）

【查询当前持仓】账号：904800028165
 当前持仓：['600829.SH', '603931.SH']（共2只）

## 问题：
1. 股池读取错误，改为9:28再读取； 
2. 卖出挂单失败，需要debug，当前日志中显示只卖出了2股，可能是代码计算的手数当成了卖出的股数，改为以股数卖出。 
3. check_position_status 函数debug功能； 
4. 第二天卖出的时间改为9:26挂卖出单； 
5. 去除全天监控是否卖出的逻辑，只在尾盘竞价前执行检查卖出情况和撤单逻辑；

6. 次日判断（只挂前一天买入的票）；
7. 撤单功能：14:55 直接查询全部委托列表，按「当前持仓代码 + 状态=未成/部成」来撤单

---

# 11月19日 

## 日志：

[2025-11-19 09:40:37][新建策略文件][SH000300][1分钟] 0D:\国投核心客户极速策略交易终端（ACT）\python\新建策略文件.py_000300129: 策略停止
[2025-11-19 09:40:38][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-19 09:40:38][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-19 09:40:38
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 卖出规则：次日9:30开盘价×1.091挂单，14:55未卖撤单，14:57跌停价卖出
================================================================================

[2025-11-19 09:40:38][新建策略文件][SH000300][1分钟] 
========================================
早盘卖出流程启动  时间：09:40:38
========================================

【查询当前持仓】账号：904800028165
 当前持仓：['600829.SH', '002083.SZ']（共2只）
【早盘卖出】600829.SH 基准价=17.72 * 1.091 -> 卖出价=19.33元

【卖出挂单】股票：600829.SH | 价格=19.33元 | 股数=200
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM583426
 卖出挂单成功 | 委托ID=MM583426
【早盘卖出】002083.SZ 基准价=12.46 * 1.091 -> 卖出价=13.59元

【卖出挂单】股票：002083.SZ | 价格=13.59元 | 股数=600
【获取最新委托ID】账号：904800028165

[2025-11-19 09:40:38][新建策略文件][SH000300][1分钟]  成功获取最新委托ID：MM583426
 卖出挂单成功 | 委托ID=MM583426
【早盘卖出】批量挂单结束

## 问题：

1. 卖出价格应该基于last close计算涨停价格，而不是开盘价。（已修复）
2. 9:28 买入时，MongoDB数据依然错误，直到29:xx，才买入新股；
3. *卖出撤单失败，测试撤单


order 字段：['-1', '0', '1', '101', '107', '108', '109', '110', '111', '112', '113', '114', '2', '255', '32', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '66', '67', '68', '69', '70', '71', '72', '73', '75', '77', '78', '79', '80', '81', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', 'EBrokerPriceType', 'ECoveredFlag', 'EEntrustBS', 'EEntrustStatus', 'EEntrustSubmitStatus', 'EEntrustTypes', 'EFutureTradeType', 'EHedge_Flag_Type', 'EOffset_Flag_Type', 'ESideFlag', 'EXTCompactBrushSource', 'EXTCompactStatus', 'EXTCompactType', 'EXTSloTypeQueryMode', 'EXTSubjectsStatus', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__instance_size__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'j', 'm_bEnable', 'm_dCancelAmount', 'm_dFrozenCommission', 'm_dFrozenMargin', 'm_dLimitPrice', 'm_dOrderPriceRMB', 'm_dReferenceRate', 'm_dShortOccupedMargin', 'm_dTradeAmount', 'm_dTradeAmountRMB', 'm_dTradedPrice', 'm_eCashgroupProp', 'm_eCoveredFlag', 'm_eEntrustType', 'm_nDirection', 'm_nErrorID', 'm_nFrontID', 'm_nGroupId', 'm_nHedgeFlag', 'm_nOffsetFlag', 'm_nOpType', 'm_nOrderPriceType', 'm_nOrderStatus', 'm_nOrderStrategyType', 'm_nOrderSubmitStatus', 'm_nRef', 'm_nSessionID', 'm_nStrategyID', 'm_nTaskId', 'm_nVolumeTotal', 'm_nVolumeTotalOriginal', 'm_nVolumeTraded', 'm_strAccountID', 'm_strAccountKey', 'm_strAccountName', 'm_strAccountRemark', 'm_strBrokerName', 'm_strCancelInfo', 'm_strCompactNo', 'm_strErrorMsg', 'm_strExchangeID', 'm_strExchangeName', 'm_strInsertDate', 'm_strInsertTime', 'm_strInstrumentID', 'm_strInstrumentName', 'm_strLocalInfo', 'm_strOptName', 'm_strOption', 'm_strOrderParam', 'm_strOrderRef', 'm_strOrderStrategyType', 'm_strOrderSysID', 'm_strProductID', 'm_strProductName', 'm_strRemark', 'm_strSource', 'm_strUnderCode', 'm_strXTTrade', 'm_xtTag', 'w', 'x', 'y', 'z']


order.m_nDirection:48
order.m_strOptName:限价卖出
order.m_nOrderStatus:54
order.m_strOrderSysID:MM582811


---

# 11月20日

## 日志：


[2025-11-20 09:06:00][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-20 09:06:00][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-20 09:06:00
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 卖出规则：次日9:30开盘价×1.091挂单，14:55未卖撤单，14:57跌停价卖出
================================================================================

[2025-11-20 09:28:01][新建策略文件][SH000300][1分钟] 
========================================
竞价买入流程启动  时间：09:28:01
========================================

【MongoDB查询】时间：2025-11-20 09:28:01（重试1/18）

[2025-11-20 09:28:01][新建策略文件][SH000300][1分钟]  获取今日股票篮子：['001216.SZ', '000632.SZ', '002474.SZ']（共3只）

【资金查询】账号：904800028165
 可用资金：9086.65元
 单只股票可用资金：3019.80元（含手续费预留）

【竞价价格查询】股票：001216.SZ
集合竞价数据-001216.SZ：开盘价=22.150000000000002, 最新价=22.150000000000002 成交量=48574, 成交额=107591400.0
 集合竞价价格：22.15元
 买入参数：001216.SZ | 价格22.59元 | 100股

【竞价价格查询】股票：000632.SZ
集合竞价数据-000632.SZ：开盘价=6.8, 最新价=6.8 成交量=551931, 成交额=375313200.0
 集合竞价价格：6.80元
 买入参数：000632.SZ | 价格6.93元 | 400股

【竞价价格查询】股票：002474.SZ
集合竞价数据-002474.SZ：开盘价=11.1, 最新价=11.1 成交量=96660, 成交额=107292600.0
 集合竞价价格：11.10元
 买入参数：002474.SZ | 价格11.32元 | 200股

[买入流程] 001216.SZ | 目标价=22.59 | lots=100

【挂单操作】股票：001216.SZ
 挂单请求发送成功：001216.SZ | 价格22.59元 | 100手

[2025-11-20 09:28:04][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：001216.SZ
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM889936
 委托校验通过 | 委托ID=MM889936 | 状态=已报（等待成交） | 股票=001216 | 价格=22.59元 | 手数=100

【持仓状态检查】股票：001216.SZ
 未找到目标持仓

[2025-11-20 09:28:05][新建策略文件][SH000300][1分钟] 
[买入流程] 000632.SZ | 目标价=6.93 | lots=400

【挂单操作】股票：000632.SZ
 挂单请求发送成功：000632.SZ | 价格6.93元 | 400手

[2025-11-20 09:28:08][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：000632.SZ
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM889938
 委托校验通过 | 委托ID=MM889938 | 状态=已报（等待成交） | 股票=000632 | 价格=6.93元 | 手数=400

【持仓状态检查】股票：000632.SZ
 未找到目标持仓

[2025-11-20 09:28:09][新建策略文件][SH000300][1分钟] 
[买入流程] 002474.SZ | 目标价=11.32 | lots=200

【挂单操作】股票：002474.SZ
 挂单请求发送成功：002474.SZ | 价格11.32元 | 200手

[2025-11-20 09:28:12][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：002474.SZ
【获取最新委托ID】账号：904800028165
 成功获取最新委托ID：MM889942
 委托校验通过 | 委托ID=MM889942 | 状态=已报（等待成交） | 股票=002474 | 价格=11.32元 | 手数=200

【持仓状态检查】股票：002474.SZ
 未找到目标持仓

[2025-11-20 09:28:13][新建策略文件][SH000300][1分钟] 【竞价买入】所有股票买入流程结束