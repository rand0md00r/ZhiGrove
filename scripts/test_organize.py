#!/usr/bin/env python3
"""
测试自动整理功能的脚本
"""

import os
import shutil
import datetime
from pathlib import Path

def get_file_creation_time(file_path):
    """获取文件创建时间（北京时间）"""
    try:
        # 设置时区为北京时间
        os.environ['TZ'] = 'Asia/Shanghai'
        
        # 获取文件状态
        stat = os.stat(file_path)
        
        # 尝试获取创建时间，回退到修改时间
        if hasattr(stat, 'st_birthtime'):  # macOS
            create_time = stat.st_birthtime
        elif hasattr(stat, 'st_ctime'):   # Linux/Windows
            create_time = stat.st_ctime
        else:
            create_time = stat.st_mtime
        
        # 转换为datetime对象（北京时间）
        dt = datetime.datetime.fromtimestamp(create_time)
        
        # 格式化为 YYMMDD-HHMM
        return dt.strftime('%y%m%d-%H%M')
        
    except Exception as e:
        print(f"   ⚠️  无法获取文件时间: {e}")
        # 使用当前北京时间作为回退
        os.environ['TZ'] = 'Asia/Shanghai'
        return datetime.datetime.now().strftime('%y%m%d-%H%M')

def organize_inbox_simulation():
    """模拟收件箱整理过程"""
    print("🔄 开始模拟收件箱自动整理...")
    
    # 设置时区为北京时间
    os.environ['TZ'] = 'Asia/Shanghai'
    
    # 获取当前日期和周次信息（北京时间）
    current_date = datetime.datetime.now()
    current_week = current_date.isocalendar()[1]
    
    print(f"📅 当前日期（北京时间）: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 当前周次: {current_week}")
    
    # 创建周目录
    week_dir = f"00-inbox/week-{current_week:02d}"
    Path(week_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 周目录: {week_dir}")
    
    # 查找收件箱中的文件
    inbox_dir = "00-inbox"
    if not os.path.exists(inbox_dir):
        print("❌ 收件箱目录不存在")
        return
    
    # 处理收件箱中的文件
    processed_count = 0
    for file_path in Path(inbox_dir).glob("*.md"):
        if file_path.name == "README.md":
            print(f"📄 {file_path.name} - ⏭️  跳过README.md")
            continue
        
        # 跳过周次目录
        if file_path.name.startswith("week-") and file_path.is_dir():
            print(f"📄 {file_path.name} - ⏭️  跳过周次目录")
            continue
        
        # 跳过已经在周次目录中的文件
        if file_path.name.startswith("[") and "]." in file_path.name:
            print(f"📄 {file_path.name} - ⏭️  跳过已整理的文件")
            continue
        
        print(f"📄 处理文件: {file_path.name}")
        
        # 获取创建时间（北京时间）
        create_time = get_file_creation_time(file_path)
        
        # 提取主题（去掉.md扩展名）
        topic = file_path.stem
        
        # 生成新文件名
        new_filename = f"[{create_time}]{topic}.md"
        new_filepath = Path(week_dir) / new_filename
        
        print(f"   📝  原文件名: {file_path.name}")
        print(f"   🏷️  主题: {topic}")
        print(f"   🕐  时间: {create_time}")
        print(f"   ✨  新文件名: {new_filename}")
        
        # 模拟移动文件（实际不移动，只显示信息）
        print(f"   📁  目标路径: {new_filepath}")
        print(f"   ✅  模拟移动完成")
        print()
        
        processed_count += 1
    
    print("📊 整理完成统计:")
    print(f"   - 处理文件数: {processed_count}")
    print(f"   - 目标目录: {week_dir}")
    
    if processed_count > 0:
        print("💡 提示: 这是模拟运行，实际文件未移动")
        print("💡 在GitHub Actions中，文件会自动移动到对应目录")
    else:
        print("ℹ️  没有文件需要整理")

def show_week_calculation():
    """显示周次计算（北京时间）"""
    print("📅 周次计算示例（北京时间）:")
    print("=" * 50)
    
    # 设置时区为北京时间
    os.environ['TZ'] = 'Asia/Shanghai'
    
    # 计算几个示例日期
    test_dates = [
        "2025-08-18",  # 周一
        "2025-08-19",  # 周二
        "2025-08-20",  # 周三
        "2025-08-21",  # 周四
        "2025-08-22",  # 周五
        "2025-08-23",  # 周六
        "2025-08-24",  # 周日
    ]
    
    for date_str in test_dates:
        try:
            year, month, day = map(int, date_str.split('-'))
            date_obj = datetime.date(year, month, day)
            year_num, week_num, weekday_num = date_obj.isocalendar()
            
            weekday_names = ["", "周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            print(f"{date_str} ({weekday_names[weekday_num]}) -> 第{week_num}周")
            
        except Exception as e:
            print(f"{date_str} -> 计算错误: {e}")
    
    print("=" * 50)
    
    # 显示当前时间信息
    now = datetime.datetime.now()
    print(f"🕐 当前时间（北京时间）: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 当前周次: {now.isocalendar()[1]}")

def main():
    print("🧪 收件箱自动整理功能测试")
    print("=" * 50)
    
    # 显示周次计算
    show_week_calculation()
    print()
    
    # 模拟整理过程
    organize_inbox_simulation()
    
    print("\n🎯 功能说明:")
    print("1. 每天早上6点自动运行（北京时间）")
    print("2. 将收件箱中的临时md文件按周次整理")
    print("3. 文件名格式: [YYMMDD-HHMM]topic.md（使用文件创建时间）")
    print("4. 目录结构: 00-inbox/week-XX/")
    print("5. 自动创建周目录（如果不存在）")
    print("6. 支持手动触发测试")
    print("7. 使用北京时间作为基准时区")

if __name__ == "__main__":
    main()
