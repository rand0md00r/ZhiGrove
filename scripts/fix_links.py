import os
import re
from urllib.parse import unquote, quote

def fix_internal_links(root_dir):
    print(f"开始扫描目录: {root_dir} 中的失效链接 (支持 URL 解码)...")
    count = 0
    
    # 匹配 Markdown 链接: [text](path)
    # 优化正则：允许 URL 中包含非右括号字符
    link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')

    def sanitize_filename_in_path(path):
        # 忽略网络链接和锚点
        if path.startswith(('http', 'https', '#', 'mailto:')):
            return path
            
        # 1. 先进行 URL 解码 (处理 %5B 这种情况)
        try:
            decoded_path = unquote(path)
        except:
            decoded_path = path

        # 分离目录和文件名
        dirname, basename = os.path.split(decoded_path)
        
        # 检查文件名是否包含旧格式的括号 (无论是否编码过)
        # 我们要找的是 [ 和 ]，或者它们被解码出来的样子
        if '[' in basename or ']' in basename:
            # 应用重命名逻辑
            # 1. 去掉 [
            # 2. 将 ] 替换为 -
            new_basename = basename.replace('[', '').replace(']', '-')
            
            # 清理
            new_basename = new_basename.replace('--', '-')
            if new_basename.endswith('-.md'):
                new_basename = new_basename.replace('-.md', '.md')
            
            # 重新组合路径
            # 注意：VitePress 通常支持未编码的中文/特殊字符路径，
            # 为了保险起见，我们直接写入可读性更好的未编码路径。
            # 只要不含空格，通常没问题；含空格 VScode 等工具会自动处理，
            # 但这里我们主要解决文件名变更。
            return os.path.join(dirname, new_basename)
        
        return path

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.md'):
                continue
                
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            def replace_callback(match):
                text = match.group(1)
                url = match.group(2)
                
                # 去除 URL 可能包含的首尾空格
                url = url.strip()
                
                # 尝试修复
                new_url = sanitize_filename_in_path(url)
                
                if new_url != url:
                    # 如果原 URL 是编码过的（%5B），new_url 是解码并修复后的（正常字符）
                    # 直接替换，通常这更易读且 VitePress 支持
                    return f'[{text}]({new_url})'
                return match.group(0)

            # 执行替换
            new_content = link_pattern.sub(replace_callback, content)

            if new_content != original_content:
                print(f"🔗 修正链接: {filename}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                count += 1

    print(f"\n链接修复完成! 共修改了 {count} 个文件。")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, 'docs')
    fix_internal_links(docs_dir)