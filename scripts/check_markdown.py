#!/usr/bin/env python3
"""
本地Markdown格式检查脚本
帮助在推送前发现格式问题
"""

import os
import glob
import re
from pathlib import Path

def check_markdown_file(file_path):
    """检查单个Markdown文件的格式"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 检查基本格式问题
        for i, line in enumerate(lines, 1):
            line_num = i
            line_content = line.rstrip('\n')
            
            # 检查行长度（超过120字符）
            if len(line_content) > 120:
                issues.append(f"  行 {line_num}: 行过长 ({len(line_content)} 字符)")
            
            # 检查尾随空格
            if line_content.endswith(' '):
                issues.append(f"  行 {line_num}: 尾随空格")
            
            # 检查制表符
            if '\t' in line_content:
                issues.append(f"  行 {line_num}: 使用制表符，建议使用空格")
        
        # 检查文件结构
        if lines:
            # 检查文件开头是否有标题
            first_line = lines[0].strip()
            if first_line and not first_line.startswith('#'):
                issues.append("  文件开头缺少标题")
            
            # 检查是否有空行结尾
            if lines and lines[-1].strip():
                issues.append("  文件末尾缺少空行")
        
        return issues
        
    except Exception as e:
        return [f"  读取文件失败: {e}"]

def main():
    print("🔍 Markdown格式检查")
    print("=" * 50)
    
    # 查找所有markdown文件
    md_files = glob.glob("**/*.md", recursive=True)
    
    if not md_files:
        print("❌ 未找到Markdown文件")
        return
    
    print(f"📁 找到 {len(md_files)} 个Markdown文件")
    print()
    
    total_issues = 0
    files_with_issues = 0
    
    for file_path in sorted(md_files):
        # 跳过.git目录
        if '.git' in file_path:
            continue
            
        issues = check_markdown_file(file_path)
        
        if issues:
            print(f"⚠️  {file_path}")
            for issue in issues:
                print(issue)
            print()
            total_issues += len(issues)
            files_with_issues += 1
        else:
            print(f"✅  {file_path}")
    
    print("=" * 50)
    print(f"📊 检查结果:")
    print(f"  总文件数: {len(md_files)}")
    print(f"  问题文件数: {files_with_issues}")
    print(f"  总问题数: {total_issues}")
    
    if total_issues == 0:
        print("🎉 所有文件格式正确！")
    else:
        print("💡 建议修复上述问题后再推送")
        print("💡 或者使用 .markdownlint.json 配置忽略某些规则")

if __name__ == "__main__":
    main()
