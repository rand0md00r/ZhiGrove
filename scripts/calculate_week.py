#!/usr/bin/env python3
"""
计算周次的脚本
"""

import datetime

def calculate_week_info(date_str):
    """计算指定日期的周次信息"""
    try:
        # 解析日期字符串 "2025-08-18"
        year, month, day = map(int, date_str.split('-'))
        target_date = datetime.date(year, month, day)
        
        # 获取ISO周次信息
        year_num, week_num, weekday_num = target_date.isocalendar()
        
        print(f"📅 日期: {target_date}")
        print(f"📊 {year_num}年第{week_num}周")
        print(f"📅 是周{weekday_num} (1=周一, 7=周日)")
        
        # 计算该周的开始和结束日期
        start_of_week = target_date - datetime.timedelta(days=weekday_num-1)
        end_of_week = start_of_week + datetime.timedelta(days=6)
        
        print(f"📅 本周范围: {start_of_week} 到 {end_of_week}")
        
        return week_num
        
    except Exception as e:
        print(f"❌ 日期解析错误: {e}")
        return None

if __name__ == "__main__":
    # 计算8月18日的周次
    week_num = calculate_week_info("2025-08-18")
    
    if week_num:
        print(f"\n✅ 结论: 2025年8月18日是第{week_num}周")
        print(f"📁 建议目录名: week-{week_num:02d}")
