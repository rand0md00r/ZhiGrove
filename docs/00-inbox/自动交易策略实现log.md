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
# MongoDB配置（请替换为你的实际配置）
MONGO_URI = "mongodb://dyx:qx_dyx@192.168.1.142:27017/?authSource=stock"
DB_NAME = "stock"
COLLECTION_NAME = "ths_realtime_stocks"
DATA_FILTER = {"phase": "竞价", "indicator_name": "千竞一号"}  # 策略筛选条件

# 交易配置
TRADE_ACCOUNT = "904800028165"  # 你的实盘账号
PRICE_MULTIPLE = 1.02          # 竞价价×1.02作为买入价
MIN_PRICE_STEP = 0.01          # 价格向下取整到0.01元
SYNC_DELAY = 3                 # 数据同步等待时间（秒）
MAX_TRADE_STOCKS = 5           # 最大买入股票数量（防止资金分散过多）

# 全局状态
IS_TRADED = False              # 当天是否已执行交易

# ===================== 2. MongoDB工具函数 =====================
import time

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
                # 后续逻辑不变
                stock_codes = doc.get("filter_code_list", [])
                if not isinstance(stock_codes, list):
                    stock_codes = []
                
                stock_codes = list(set(stock_codes))
                if len(stock_codes) > MAX_TRADE_STOCKS:
                    stock_codes = stock_codes[:MAX_TRADE_STOCKS]
                    print(f" 股票数量超过{MAX_TRADE_STOCKS}只，仅买入前{MAX_TRADE_STOCKS}只")
                
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
        print(f"\n【竞价价查询】股票：{stock_code}")
        # 旧版QMT获取竞价数据（字段用"open"，集合竞价时段返回竞价价）
        # data = ContextInfo.get_market_data(
        #     fields=["open"],
        #     stock_code=[stock_code],
        #     period="1d",
        #     dividend_type="front",
        #     count=1
        # )

        full_tick = ContextInfo.get_full_tick([stock_code])
        
        # # 解析数据（兼容不同返回格式）
        # auction_price = 0.0
        # if isinstance(data, dict) and stock_code in data:
        #     auction_price = data[stock_code]["open"]
        # elif hasattr(data, "iloc") and not data.empty:
        #     auction_price = data["open"].iloc[-1]
        # elif isinstance(data, list) and len(data) > 0:
        #     auction_price = data[0]

        last_price = full_tick[stock_code]["lastPrice"]
        open_price = tick_data[stock_code]['open']  # 集合竞价最终开盘价
        volume = tick_data[stock_code]['volume']    # 集合竞价成交量
        amount = tick_data[stock_code]['amount']    # 集合竞价成交额

        print(f"集合竞价数据-{stock_code}：开盘价={open_price}, 最新价={last_price} 成交量={volume}, 成交额={amount}")

        auction_price = open_price

        # 价格校验
        if auction_price <= 0 or math.isnan(auction_price):
            print(f"? 竞价价异常：{auction_price}元（可能非竞价时段）")
            return 0.0
        
        print(f"? 集合竞价价格：{auction_price:.2f}元")
        return auction_price
    
    except Exception as e:
        print(f"? 竞价价查询异常：{str(e)}")
        return 0.0

def calculate_buy_params(ContextInfo, stock_codes):
    """计算每个股票的买入价格和手数（按资金均分）"""
    buy_params = []
    try:
        # 1. 查询账户可用资金
        print(f"\n【资金查询】账号：{TRADE_ACCOUNT}")
        acct_data = get_trade_detail_data(TRADE_ACCOUNT, "stock", "account")

        if not isinstance(acct_data, list) or len(acct_data) == 0:
            print(f"? 未查询到账户资金信息")
            return []
        
        # 提取可用资金（旧版字段：m_dAvailable）
        available_cash = 0.0
        for acct in acct_data:
            if hasattr(acct, "m_dAvailable"):
                available_cash = acct.m_dAvailable
                break
        
        if available_cash <= 100:  # 最低交易门槛
            print(f"? 可用资金不足：{available_cash:.2f}元")
            return []
        
        print(f"? 可用资金：{available_cash:.2f}元")
        
        # 2. 计算单只股票可用资金（均分+预留手续费）
        stock_count = len(stock_codes)
        if stock_count == 0:
            return []
        
        # 预留千分之3手续费，实际可用资金=总资金×(1-手续费率)/股票数量
        commission_rate = 0.003
        single_stock_cash = available_cash * (1 - commission_rate) / stock_count
        print(f"? 单只股票可用资金：{single_stock_cash:.2f}元（含手续费预留）")
        
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
            
            print(f"? 买入参数：{stock} | 价格{buy_price:.2f}元 | {max_lots}手（{max_lots*100}股）")
        
        return buy_params
    
    except Exception as e:
        print(f"? 计算买入参数异常：{str(e)}")
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
        print(f"? 挂单失败：{str(e)}")
        return False

def check_order_status(stock_code, expected_price, expected_lots):
    """检查委托状态（判断是否成功挂单）"""
    try:
        print(f"\n【委托状态检查】股票：{stock_code}")
        # 查询委托列表
        order_data = get_trade_detail_data(TRADE_ACCOUNT, "stock", "order")
        
        if not isinstance(order_data, list) or len(order_data) == 0:
            print(f" 未查询到委托记录")
            return False
        
        # 筛选目标委托（匹配股票代码、价格、手数，且状态为未成交/部分成交）
        target_order = None
        for order in order_data:
            # 旧版委托字段：m_strOrderCode(股票代码), m_dModelPrice(委托价), m_nVolume(手数), m_nOrderStatus(状态)
            if (hasattr(order, "m_strOrderCode") and order.m_strOrderCode == stock_code and
                hasattr(order, "m_dModelPrice") and math.isclose(order.m_dModelPrice, expected_price) and
                hasattr(order, "m_nVolume") and order.m_nVolume == expected_lots and
                hasattr(order, "m_nOrderStatus") and order.m_nOrderStatus in (50, 51)):  # 50=未成交，51=部分成交
                target_order = order
                break
        
        if target_order:
            print(f"? 委托成功：委托号{target_order.m_nTaskId} | 状态{target_order.m_nOrderStatus}")
            return True
        else:
            # 打印所有委托记录，方便排查
            print(f" 未找到目标委托，当前所有委托：")
            for idx, order in enumerate(order_data):
                print(f"  委托{idx+1}：{order.m_strOrderCode} | {order.m_dModelPrice}元 | {order.m_nVolume}手 | 状态{order.m_nOrderStatus}")
            return False
    
    except Exception as e:
        print(f"? 委托状态查询异常：{str(e)}")
        return False

def check_position_status(stock_code, expected_lots):
    """检查持仓状态（判断是否成功买入）"""
    try:
        print(f"\n【持仓状态检查】股票：{stock_code}")
        # 查询持仓列表
        position_data = get_trade_detail_data(TRADE_ACCOUNT, "stock", "position")
        
        if not isinstance(position_data, list) or len(position_data) == 0:
            print(f" 未查询到持仓记录")
            return False
        
        # 筛选目标持仓（匹配股票代码，且持仓数量≥预期手数×100）
        target_position = None
        expected_shares = expected_lots * 100  # 预期股数
        for pos in position_data:
            # 旧版持仓字段：m_strStockCode(股票代码), m_nVolume(持仓股数)
            if (hasattr(pos, "m_strStockCode") and pos.m_strStockCode == stock_code and
                hasattr(pos, "m_nVolume") and pos.m_nVolume >= expected_shares):
                target_position = pos
                break
        
        if target_position:
            print(f"? 买入成功：{stock_code} | 持仓{target_position.m_nVolume}股 | 成本{target_position.m_dCostPrice:.2f}元")
            return True
        else:
            # 打印所有持仓记录，方便排查
            print(f" 未找到目标持仓，当前所有持仓：")
            for idx, pos in enumerate(position_data):
                print(f"  持仓{idx+1}：{pos.m_strStockCode} | {pos.m_nVolume}股 | 成本{pos.m_dCostPrice:.2f}元")
            return False
    
    except Exception as e:
        print(f"? 持仓状态查询异常：{str(e)}")
        return False

# ===================== 5. QMT策略核心函数 =====================
def init(ContextInfo):
    """策略初始化（仅执行一次）"""
    global IS_TRADED
    IS_TRADED = False
    
    # 绑定交易账号
    ContextInfo.accID = TRADE_ACCOUNT
    ContextInfo.set_account(TRADE_ACCOUNT)
    
    print("="*80)
    print(f"【实盘买入策略启动】时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置信息：")
    print(f"  - MongoDB：{MONGO_URI.split('@')[-1]}/{DB_NAME}/{COLLECTION_NAME}")
    print(f"  - 交易账号：{TRADE_ACCOUNT}")
    print(f"  - 买入规则：竞价价×{PRICE_MULTIPLE}（向下取整），资金均分")
    print(f"  - 最大买入股票：{MAX_TRADE_STOCKS}只")
    print("="*80)

def handlebar(ContextInfo):
    """策略循环执行（仅在竞价期9:27-9:30执行）"""
    global IS_TRADED
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # 执行条件：① 未交易过 ② 竞价时段（9:27-9:30）
    if IS_TRADED or not ("09:27:00" <= current_time <= "09:30:00"):
       return
    # if IS_TRADED:
    #     return

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
        print(f"\n? 所有股票买入操作已完成")
    
    except Exception as e:
        print(f"\n? 策略执行异常：{str(e)}")
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

