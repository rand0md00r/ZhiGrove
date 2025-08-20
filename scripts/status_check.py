#!/usr/bin/env python3
"""
状态检查脚本 - 检查知识库的健康状态
"""

import os
import glob
from datetime import datetime, timedelta
from pathlib import Path

def count_files(directory, pattern="*.md"):
    """统计目录中的文件数量"""
    if not os.path.exists(directory):
        return 0
    return len(glob.glob(os.path.join(directory, pattern)))

def get_recent_files(directory, days=7, pattern="*.md"):
    """获取最近几天的文件"""
    if not os.path.exists(directory):
        return []
    
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_files = []
    
    for file_path in glob.glob(os.path.join(directory, pattern)):
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        if mtime > cutoff_date:
            recent_files.append((file_path, mtime))
    
    return sorted(recent_files, key=lambda x: x[1], reverse=True)

def check_inbox_status():
    """检查收件箱状态"""
    inbox_dir = "00-inbox"
    total_files = count_files(inbox_dir)
    recent_files = get_recent_files(inbox_dir, days=7)
    
    print("📥 收件箱状态")
    print(f"   总文件数: {total_files}")
    print(f"   最近7天: {len(recent_files)}")
    
    if total_files > 20:
        print("   ⚠️  收件箱文件过多，建议及时整理")
    elif total_files > 10:
        print("   ⚠️  收件箱文件较多，建议本周整理")
    else:
        print("   ✅ 收件箱状态良好")
    
    if recent_files:
        print("   最近文件:")
        for file_path, mtime in recent_files[:3]:
            name = os.path.basename(file_path)
            print(f"     - {name} ({mtime.strftime('%m-%d %H:%M')})")
    print()

def check_knowledge_status():
    """检查知识库状态"""
    knowledge_dir = "10-knowledge"
    total_files = count_files(knowledge_dir)
    
    print("📚 知识库状态")
    print(f"   总文件数: {total_files}")
    
    if total_files == 0:
        print("   ⚠️  知识库为空，建议开始整理已有内容")
    elif total_files < 5:
        print("   ⚠️  知识库内容较少，建议增加知识沉淀")
    else:
        print("   ✅ 知识库状态良好")
    print()

def check_papers_status():
    """检查论文笔记状态"""
    papers_dir = "20-papers"
    total_files = count_files(papers_dir, "**/*.md")
    recent_files = get_recent_files(papers_dir, days=30, pattern="**/*.md")
    
    print("📄 论文笔记状态")
    print(f"   总文件数: {total_files}")
    print(f"   最近30天: {len(recent_files)}")
    
    if total_files == 0:
        print("   ⚠️  论文笔记为空，建议开始记录论文学习")
    elif len(recent_files) == 0:
        print("   ⚠️  最近没有新的论文笔记，建议保持学习节奏")
    else:
        print("   ✅ 论文笔记状态良好")
    print()

def check_ideas_status():
    """检查想法记录状态"""
    ideas_dir = "30-ideas"
    total_files = count_files(ideas_dir, "**/*.md")
    recent_files = get_recent_files(ideas_dir, days=7, pattern="**/*.md")
    
    print("💡 想法记录状态")
    print(f"   总文件数: {total_files}")
    print(f"   最近7天: {len(recent_files)}")
    
    if total_files == 0:
        print("   ⚠️  想法记录为空，建议开始记录灵感")
    elif len(recent_files) == 0:
        print("   ⚠️  最近没有新的想法记录，建议保持记录习惯")
    else:
        print("   ✅ 想法记录状态良好")
    print()

def check_experiments_status():
    """检查实验记录状态"""
    experiments_dir = "40-experiments"
    total_dirs = len([d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d)) and d.startswith("exp-")])
    recent_dirs = []
    
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        if os.path.isdir(exp_path) and exp_dir.startswith("exp-"):
            mtime = datetime.fromtimestamp(os.path.getmtime(exp_path))
            if mtime > datetime.now() - timedelta(days=30):
                recent_dirs.append((exp_dir, mtime))
    
    print("🔬 实验记录状态")
    print(f"   总实验数: {total_dirs}")
    print(f"   最近30天: {len(recent_dirs)}")
    
    if total_dirs == 0:
        print("   ⚠️  实验记录为空，建议开始记录实验过程")
    elif len(recent_dirs) == 0:
        print("   ⚠️  最近没有新的实验记录，建议保持实验记录")
    else:
        print("   ✅ 实验记录状态良好")
    print()

def check_reports_status():
    """检查报告状态"""
    reports_dir = "50-reports"
    weekly_dir = os.path.join(reports_dir, "weekly")
    monthly_dir = os.path.join(reports_dir, "monthly")
    
    weekly_files = count_files(weekly_dir, "**/*.md")
    monthly_files = count_files(monthly_dir, "**/*.md")
    
    print("📊 报告状态")
    print(f"   周报数量: {weekly_files}")
    print(f"   月报数量: {monthly_files}")
    
    if weekly_files == 0 and monthly_files == 0:
        print("   ⚠️  报告记录为空，建议开始定期总结")
    elif weekly_files < 4:
        print("   ⚠️  周报数量较少，建议保持周报习惯")
    else:
        print("   ✅ 报告状态良好")
    print()

def check_overall_health():
    """检查整体健康状态"""
    print("🏥 整体健康检查")
    print("=" * 50)
    
    check_inbox_status()
    check_knowledge_status()
    check_papers_status()
    check_ideas_status()
    check_experiments_status()
    check_reports_status()
    
    print("📋 建议行动")
    print("-" * 30)
    
    # 基于检查结果给出建议
    inbox_count = count_files("00-inbox")
    if inbox_count > 15:
        print("1. 🚨 优先整理收件箱，清理超过15个文件")
    
    knowledge_count = count_files("10-knowledge")
    if knowledge_count < 3:
        print("2. 📚 开始整理已有内容到知识库")
    
    papers_count = count_files("20-papers", "**/*.md")
    if papers_count == 0:
        print("3. 📄 开始记录论文学习笔记")
    
    ideas_count = count_files("30-ideas", "**/*.md")
    if ideas_count < 5:
        print("4. 💡 增加想法和灵感的记录")
    
    experiments_count = len([d for d in os.listdir("40-experiments") if os.path.isdir(os.path.join("40-experiments", d)) and d.startswith("exp-")])
    if experiments_count == 0:
        print("5. 🔬 开始记录实验过程和结果")
    
    print("\n💡 提示：使用 'python scripts/quick_note.py' 快速记录")

def main():
    check_overall_health()

if __name__ == "__main__":
    main() 