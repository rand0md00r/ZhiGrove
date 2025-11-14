挂单策略流程图：

1. 读取MongoDB篮子股票信息；
2. 计算每个股票的买单信息：按价格买入（1102），计算买入价格（竞价末尾位置的1.02，向下取整）；
3. 挂单；
4. 检查委托列表，判断是否成功挂单；
5. 检查买入列表，判断是否成功买入；


#########################################################################################################################

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
def get_market_code(stock_code):
    """获取市场代码（int类型，适配passorder）：上海=1，深圳=2"""
    if stock_code.endswith(".SH") or stock_code.startswith("60"):
        return 1, "上海市场"
    elif stock_code.endswith(".SZ") or stock_code.startswith(("00", "30")):
        return 2, "深圳市场"
    else:
        print(f" 未知市场：{stock_code}，默认上海市场（1）")
        return 1, "默认上海市场"

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
            max_lots = int(max_shares // 100) * 100
            
            if max_lots < 100:
                print(f" 资金不足买入1手：{stock}（需{buy_price*100:.2f}元，可用{single_stock_cash:.2f}元）")
                continue
            
            # 记录买入参数
            market_code, _ = get_market_code(stock)
            buy_params.append({
                "stock_code": stock,
                "buy_price": buy_price,
                "lots": max_lots,
                "market_code": market_code
            })
            
            print(f" 买入参数：{stock} | 价格{buy_price:.2f}元 | {max_lots}手（{max_lots*100}股）")
        
        return buy_params
    
    except Exception as e:
        print(f" 计算买入参数异常：{str(e)}")
        return []

# ===================== 4. 挂单与状态检查函数 =====================
def place_order(ContextInfo, stock_code, buy_price, lots, market_code):
    """执行挂单操作（下单类型1102=按价格买入）"""
    try:
        print(f"\n【挂单操作】股票：{stock_code}")
        # passorder参数（旧版纯位置传参，共10个参数）
        # 参数顺序：opType(23=买入), orderType(1102=按价格), 账号, 股票代码, prType(11=指定价), 
        #          挂单价格, 手数(double), 市场代码(int), 策略名称, 消息, ContextInfo
        print(f"market_code: {market_code}")
        print(f"float(lots): {float(lots)}")
        passorder(23, 1101, TRADE_ACCOUNT, stock_code, 11, buy_price, float(lots), '千竞策略', 2, "实盘买入", ContextInfo)
        print(f" 挂单请求发送成功：{stock_code} | 价格{buy_price:.2f}元 | {lots}手")
        return True

    except Exception as e:
        print(f" 挂单失败：{str(e)}")
        return False

def place_sell_order(ContextInfo, stock_code, sell_price, lots):
    """执行卖出挂单（opType=24=股票卖出，返回委托ID）"""
    try:
        print(f"\n【卖出挂单】股票：{stock_code} | 价格={sell_price:.2f}元 | 手数={lots}")
        # 调用passorder卖出：opType=24（卖出），orderType=1101（按手数），prType=11（指定价）
        passorder(24, 1101, TRADE_ACCOUNT, stock_code, 11, sell_price, float(lots), '千竞策略', 2, "实盘卖出", ContextInfo)
        
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

def check_position_status(stock_code, expected_lots):
    """检查持仓状态（判断是否成功买入，新增记录持仓到ContextInfo）"""
    try:
        print(f"\n【持仓状态检查】股票：{stock_code}")
        position_data = get_trade_detail_data(TRADE_ACCOUNT, "stock", "position")
        
        if not isinstance(position_data, list) or len(position_data) == 0:
            print(f" 未查询到持仓记录")
            return False
        
        target_position = None
        expected_shares = expected_lots * 100
        for pos in position_data:
            # 优先用API标准字段m_strInstrumentID匹配
            pos_stock_code = pos.m_strInstrumentID if hasattr(pos, "m_strInstrumentID") else pos.m_strStockCode
            if (pos_stock_code == stock_code and
                hasattr(pos, "m_nVolume") and pos.m_nVolume >= expected_shares):
                target_position = pos
                break
        
        if target_position:
            cost_price = target_position.m_dOpenPrice if hasattr(target_position, "m_dOpenPrice") else 0.0
            print(f" 买入成功：{stock_code} | 持仓{target_position.m_nVolume}股 | 成本{cost_price:.2f}元")
            
            # 新增：记录持仓到ContextInfo（供第二天卖出使用）
            hold_info = {
                "stock_code": stock_code,
                "lots": expected_lots,
                "morning_order_id": None,  # 早盘卖出委托ID
                "is_sold": False  # 是否已卖出
            }
            # 避免重复添加持仓
            if not any(h["stock_code"] == stock_code for h in ContextInfo.hold_stocks):
                ContextInfo.hold_stocks.append(hold_info)
            return True
        else:
            print(f" 未找到目标持仓")
            return False
    
    except Exception as e:
        print(f" 持仓状态查询异常：{str(e)}")
        return False

# ===================== 5. QMT策略核心函数 =====================
def init(ContextInfo):
    """策略初始化（仅执行一次）"""
    global IS_TRADED
    IS_TRADED = False
    
    # 绑定交易账号
    ContextInfo.accID = TRADE_ACCOUNT
    ContextInfo.set_account(TRADE_ACCOUNT)
    
    # 新增：卖出相关全局状态（跨交易日存储，依赖ContextInfo）
    ContextInfo.hold_stocks = []  # 存储持仓股票信息：[{stock_code, lots, morning_order_id, is_sold}]
    ContextInfo.morning_sold = False  # 是否已执行早盘卖出（9:30）
    ContextInfo.afternoon_sold = False  # 是否已执行尾盘卖出（14:57）
    
    print("="*80)
    print(f"【实盘买入策略启动】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置信息：")
    print(f"  - MongoDB：{MONGO_URI.split('@')[-1]}/{DB_NAME}/{COLLECTION_NAME}")
    print(f"  - 交易账号：{TRADE_ACCOUNT}")
    print(f"  - 买入规则：竞价价×{PRICE_MULTIPLE}（向下取整），资金均分")
    print(f"  - 卖出规则：次日9:30涨停价×0.91挂单，14:55未卖撤单，14:57跌停价卖出")
    print("="*80)

def handlebar(ContextInfo):
    """策略循环执行（仅在竞价期9:27-9:30执行）"""
    global IS_TRADED
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # ===================== 新增：卖出逻辑 =====================
    # 1. 早盘卖出（第二天9:30-9:30:30，仅执行一次）
    if (not ContextInfo.morning_sold and 
        "09:30:00" <= current_time <= "09:30:30" and 
        len(ContextInfo.hold_stocks) > 0):
        print(f"\n【早盘卖出】时间窗口触发（{current_time}）")
        for hold in ContextInfo.hold_stocks:
            if hold["is_sold"]:
                continue
            stock_code = hold["stock_code"]
            lots = hold["lots"]
            
            # 获取涨停价，按涨停价×0.91挂单
            up_limit, _ = get_limit_prices(ContextInfo, stock_code)
            if up_limit <= 0:
                print(f" 跳过{stock_code}：涨停价计算异常")
                continue
            sell_price = math.floor(up_limit * 0.91 * 100) / 100  # 向下取整到0.01
            
            # 执行卖出挂单，记录委托ID
            sell_order_id = place_sell_order(ContextInfo, stock_code, sell_price, lots)
            if sell_order_id:
                hold["morning_order_id"] = sell_order_id  # 记录委托ID供撤单使用
        
        ContextInfo.morning_sold = True  # 标记已执行早盘卖出
        print(f"【早盘卖出】批量挂单完成，等待成交")
        return

    # 2. 新增：早盘成交监控（9:30:31-14:49:59，持续检查是否成交）
    elif (ContextInfo.morning_sold and 
          not ContextInfo.afternoon_sold and 
          "09:30:31" <= current_time <= "14:49:59" and 
          len(ContextInfo.hold_stocks) > 0):
        # 每30秒检查一次（避免频繁请求，可调整间隔）
        current_second = datetime.now().second
        if current_second % 30 != 0:  # 只在整30秒时检查（如9:30:30、9:31:00...）
            return
        
        print(f"\n【早盘成交监控】时间：{current_time}（每30秒检查一次）")
        for hold in ContextInfo.hold_stocks:
            # 已卖出或无委托ID，跳过
            if hold["is_sold"] or not hold["morning_order_id"]:
                continue
            
            stock_code = hold["stock_code"]
            order_id = hold["morning_order_id"]
            expected_lots = hold["lots"]
            
            # 用委托ID查询最新状态（判断是否成交）
            order_detail = get_value_by_order_id(order_id, TRADE_ACCOUNT, 'stock', 'ORDER')
            if not order_detail:
                print(f" 跳过{stock_code}：委托ID={order_id} 详情查询失败")
                continue
            
            # 检查委托状态：56=已成，55=部分成交（均视为成交）
            if hasattr(order_detail, "m_nOrderStatus") and order_detail.m_nOrderStatus in (55, 56):
                # 验证持仓是否同步（双重确认）
                if check_position_status(stock_code, expected_lots):
                    hold["is_sold"] = True
                    hold["morning_order_id"] = None  # 清空委托ID
                    print(f" 监控到{stock_code}成交 | 委托ID={order_id} | 状态={order_detail.m_nOrderStatus}")
            else:
                # 未成交，打印当前状态
                status_desc = {50:"已报",51:"已报待撤",54:"已撤",57:"废单"}.get(order_detail.m_nOrderStatus, f"未知({order_detail.m_nOrderStatus})")
                print(f" {stock_code}未成交 | 委托ID={order_id} | 状态={status_desc} | 挂单价={order_detail.m_dLimitPrice:.2f}元")
        return

    # 2. 撤单逻辑（14:55-14:55:30，仅执行一次）
    if (ContextInfo.morning_sold and 
        not ContextInfo.afternoon_sold and 
        "14:55:00" <= current_time <= "14:55:30" and 
        len(ContextInfo.hold_stocks) > 0):
        print(f"\n【尾盘撤单】时间窗口触发（{current_time}）")
        for hold in ContextInfo.hold_stocks:
            if hold["is_sold"] or not hold["morning_order_id"]:
                continue
            # 撤销早盘未成交委托
            cancel_result = cancel_order(ContextInfo, hold["morning_order_id"], hold["stock_code"])
            if cancel_result:
                hold["morning_order_id"] = None  # 清空委托ID
        print(f"【尾盘撤单】批量撤单完成，准备尾盘卖出")
        return

    # 3. 尾盘卖出（14:57-14:57:30，仅执行一次）
    if (not ContextInfo.afternoon_sold and 
        "14:57:00" <= current_time <= "14:57:30" and 
        len(ContextInfo.hold_stocks) > 0):
        print(f"\n【尾盘卖出】时间窗口触发（{current_time}）")
        for hold in ContextInfo.hold_stocks:
            if hold["is_sold"]:
                continue
            stock_code = hold["stock_code"]
            lots = hold["lots"]
            
            # 获取跌停价，挂跌停价卖出
            _, down_limit = get_limit_prices(ContextInfo, stock_code)
            if down_limit <= 0:
                print(f" 跳过{stock_code}：跌停价计算异常")
                continue
            
            # 执行尾盘卖出挂单
            place_sell_order(ContextInfo, stock_code, down_limit, lots)
            hold["is_sold"] = True  # 标记已发起尾盘卖出
        
        ContextInfo.afternoon_sold = True  # 标记已执行尾盘卖出
        print(f"【尾盘卖出】批量挂单完成")
        return

    # 执行条件：① 未交易过 ② 竞价时段（9:27-9:30）
    if not IS_TRADED and "09:27:30" <= current_time <= "09:30:00":
        try:
            # 1. 读取MongoDB股票篮子
            stock_codes = get_today_stock_basket()
            if len(stock_codes) == 0:
                IS_TRADED = True  # 标记为已交易（避免重复执行）
                return
            
            # 2. 计算买入参数（价格+手数）
            buy_params = calculate_buy_params(ContextInfo, stock_codes)
            if len(buy_params) == 0:
                print(f" 无有效买入参数，终止交易")
                IS_TRADED = True
                return
            
            # 3. 逐股执行挂单+状态检查
            for params in buy_params:
                stock_code = params["stock_code"]
                buy_price = params["buy_price"]
                lots = params["lots"]
                market_code = params["market_code"]
                
                # 挂单
                if not place_order(ContextInfo, stock_code, buy_price, lots, market_code):
                    continue
                
                # 等待数据同步
                time.sleep(SYNC_DELAY)
                
                # 检查委托状态
                if check_order_status(stock_code, buy_price, lots):
                    # 检查持仓状态（可选：如果需要确认成交，可增加循环检查）
                    check_position_status(stock_code, lots)
                
                # 每只股票操作后间隔1秒，避免请求过于频繁
                time.sleep(1)
            
            # 4. 标记为已交易（当天不再执行）
            IS_TRADED = True
            print(f"\n 所有股票买入操作已完成")
        
        except Exception as e:
            print(f"\n 策略执行异常：{str(e)}")
            IS_TRADED = True  # 避免异常后重复执行
    

def stop(ContextInfo):
    """策略停止"""
    print("\n" + "="*80)
    print(f"【策略停止】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"今日交易状态：{'已执行买入操作' if IS_TRADED else '未执行交易'}")
    print("建议：登录QMT交易模块，查看「委托记录」和「持仓」确认最终状态")
    print("="*80)

```




[2025-11-12 09:02:22][新建策略文件][SH000300][1分钟] [trade]start trading mode
[2025-11-12 09:02:22][新建策略文件][SH000300][1分钟] ================================================================================
【实盘买入策略启动】时间：2025-11-12 09:02:22
配置信息：
  - MongoDB：192.168.1.142:27017/?authSource=stock/stock/ths_realtime_stocks
  - 交易账号：904800028165
  - 买入规则：竞价价×1.02（向下取整），资金均分
  - 最大买入股票：5只
================================================================================

[2025-11-12 09:27:04][新建策略文件][SH000300][1分钟] 
【MongoDB查询】时间：2025-11-12 09:27:04

[2025-11-12 09:27:04][新建策略文件][SH000300][1分钟] ? 获取今日股票篮子：['000090.SZ', '000037.SZ']（共2只）

【资金查询】账号：904800028165
? 可用资金：9999.00元
? 单只股票可用资金：4984.50元（含手续费预留）

【竞价价查询】股票：000090.SZ
get_market_data接口版本较老，推荐使用get_market_data_ex替代，配合download_history_data补充昨日以前的历史数据

[2025-11-12 09:27:04][新建策略文件][SH000300][1分钟] ? 集合竞价价格：3.83元
? 买入参数：000090.SZ | 价格3.90元 | 1200手（120000股）

【竞价价查询】股票：000037.SZ

[2025-11-12 09:27:04][新建策略文件][SH000300][1分钟] ? 集合竞价价格：10.40元
? 买入参数：000037.SZ | 价格10.60元 | 400手（40000股）

【挂单操作】股票：000090.SZ
market_code: 2
float(lots): 1200.0
 挂单请求发送成功：000090.SZ | 价格3.90元 | 1200手

[2025-11-12 09:27:07][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：000090.SZ
 未找到目标委托，当前所有委托：
? 委托状态查询异常：'COrderDetail' object has no attribute 'm_strOrderCode'

[2025-11-12 09:27:08][新建策略文件][SH000300][1分钟] 
【挂单操作】股票：000037.SZ
market_code: 2
float(lots): 400.0
 挂单请求发送成功：000037.SZ | 价格10.60元 | 400手

[2025-11-12 09:27:11][新建策略文件][SH000300][1分钟] 
【委托状态检查】股票：000037.SZ
 未找到目标委托，当前所有委托：
? 委托状态查询异常：'COrderDetail' object has no attribute 'm_strOrderCode'

[2025-11-12 09:27:12][新建策略文件][SH000300][1分钟] 
? 所有股票买入操作已完成

[2025-11-12 09:55:05][新建策略文件][SH000300][1分钟] 0D:\国投核心客户极速策略交易终端（ACT）\python\新建策略文件.py_00030014: 策略停止
[2025-11-12 09:55:05][新建策略文件][SH000300][1分钟] 
================================================================================
【策略停止】时间：2025-11-12 09:55:05
今日交易状态：已执行买入操作
建议：登录QMT交易模块，查看「委托记录」和「持仓」确认最终状态
================================================================================


--- 2025-11-12 问题

1. 27:00 查询到的数据是错误的个股；
2. buy_price错误，







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

11月13日

1. 买入：千尾；
2. 卖出：千竞 = 涨停价格 * 0.97，向下取整； 千尾 = 涨停价格 * 0.91，向上取整； 没涨停，尾盘直接卖出；
3. 买入的有效检查 = 查委托 + 查持仓；
4. 卖出逻辑：第二天早盘9：30开始挂出卖单，按涨停价格 * 0.91卖出；下午2:55检查是否完成卖出，如果没卖出，撤单，2:57尾盘竞价开始时，挂跌停价卖出；



---


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

 