import os

def sanitize_filenames(root_dir):
    print(f"开始扫描目录: {root_dir} ...")
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件名是否包含不兼容的字符 [ 或 ]
            if '[' in filename or ']' in filename:
                old_path = os.path.join(dirpath, filename)
                
                # 新命名逻辑：
                # 1. 去掉 '['
                # 2. 将 ']' 替换为 '-'
                new_filename = filename.replace('[', '').replace(']', '-')
                
                # 清理可能产生的双横线 '--' 和末尾的 '-'
                new_filename = new_filename.replace('--', '-')
                if new_filename.endswith('-.md'):
                    new_filename = new_filename.replace('-.md', '.md')
                
                new_path = os.path.join(dirpath, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"✅ 重命名: {filename} \n   -> {new_filename}")
                    count += 1
                except OSError as e:
                    print(f"❌ 重命名失败 {filename}: {e}")

    print(f"\n处理完成! 共修改了 {count} 个文件名。")

if __name__ == "__main__":
    # 获取脚本所在目录的上级目录，再定位到 docs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, 'docs')
    
    if os.path.exists(docs_dir):
        sanitize_filenames(docs_dir)
    else:
        print(f"错误: 找不到 docs 目录 ({docs_dir})")