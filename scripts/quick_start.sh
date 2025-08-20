#!/usr/bin/env bash
# 快速启动脚本 - 常用操作的快捷方式

echo "🚀 ZhiGrove 快速启动"
echo "=================="

case "$1" in
  "status"|"s")
    echo "📊 检查知识库状态..."
    python scripts/status_check.py
    ;;
    
  "note"|"n")
    if [ -z "$2" ]; then
      echo "📝 快速记录想法..."
      echo "用法: $0 note '你的想法内容'"
      echo "或者: $0 note -t idea '想法内容'"
      echo "或者: $0 note -t paper '论文标题'"
      echo "或者: $0 note -t experiment '实验名称'"
      exit 1
    fi
    
    if [ "$2" = "-t" ]; then
      python scripts/quick_note.py "$4" -t "$3"
    else
      python scripts/quick_note.py "$2"
    fi
    ;;
    
  "idea"|"i")
    if [ -z "$2" ]; then
      echo "💡 快速记录想法..."
      echo "用法: $0 idea '你的想法'"
      exit 1
    fi
    python scripts/quick_note.py "$2" -t idea
    ;;
    
  "paper"|"p")
    if [ -z "$2" ]; then
      echo "📄 快速创建论文笔记..."
      echo "用法: $0 paper '论文标题'"
      exit 1
    fi
    python scripts/quick_note.py "$2" -t paper
    ;;
    
  "experiment"|"exp")
    if [ -z "$2" ]; then
      echo "🔬 快速创建实验记录..."
      echo "用法: $0 experiment '实验名称'"
      exit 1
    fi
    python scripts/quick_note.py "$2" -t experiment
    ;;
    
  "inbox"|"in")
    if [ -z "$2" ]; then
      echo "📥 快速记录到收件箱..."
      echo "用法: $0 inbox '内容'"
      exit 1
    fi
    python scripts/quick_note.py "$2" -t inbox
    ;;
    
  "update"|"u")
    echo "🔄 更新索引..."
    python scripts/build_index.py
    ;;
    
  "help"|"h"|"")
    echo "📚 可用命令："
    echo ""
    echo "  status, s     - 检查知识库状态"
    echo "  note, n       - 快速记录（通用）"
    echo "  idea, i       - 快速记录想法"
    echo "  paper, p      - 快速创建论文笔记"
    echo "  experiment, exp - 快速创建实验记录"
    echo "  inbox, in     - 快速记录到收件箱"
    echo "  update, u     - 更新索引"
    echo "  help, h       - 显示帮助"
    echo ""
    echo "💡 快速开始："
    echo "  $0 idea '我的新想法'"
    echo "  $0 paper '论文标题'"
    echo "  $0 status"
    ;;
    
  *)
    echo "❌ 未知命令: $1"
    echo "使用 '$0 help' 查看可用命令"
    exit 1
    ;;
esac 