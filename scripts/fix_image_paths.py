import os
import re

def fix_image_paths(root_dir):
    print(f"开始扫描目录: {root_dir} 中的图片路径问题...")
    count = 0
    
    # --- 规则 1: 标准 assets 目录 ---
    # 目标: 将 assets/..., ./assets/..., ../assets/... 替换为 /assets/...
    md_pattern = r'\]\(\s*(?:\./|\.\./)*assets/'
    html_pattern = r'src=["\']\s*(?:\./|\.\./)*assets/'

    # --- 规则 2: vla_assets 目录 (特例) ---
    # 目标: 将 vla_assets/..., ./vla_assets/... 替换为 /00-inbox/vla_assets/...
    # 注意: 你的 vla_assets 实际位于 docs/00-inbox/vla_assets，需要使用该绝对路径
    vla_md_pattern = r'\]\(\s*(?:\./|\.\./)*vla_assets/'
    vla_html_pattern = r'src=["\']\s*(?:\./|\.\./)*vla_assets/'

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.md'):
                continue
                
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # --- 执行 assets 替换 ---
            # 1. Markdown
            content = re.sub(md_pattern, '](/assets/', content)
            
            # 2. HTML
            def replace_html(match):
                text = match.group(0)
                quote = '"' if '"' in text else "'"
                return f'src={quote}/assets/'
            content = re.sub(html_pattern, replace_html, content)

            # --- 执行 vla_assets 替换 ---
            # 3. Markdown (vla_assets)
            content = re.sub(vla_md_pattern, '](/00-inbox/vla_assets/', content)

            # 4. HTML (vla_assets)
            def replace_vla_html(match):
                text = match.group(0)
                quote = '"' if '"' in text else "'"
                return f'src={quote}/00-inbox/vla_assets/'
            content = re.sub(vla_html_pattern, replace_vla_html, content)

            if content != original_content:
                print(f"🖼️  修正路径: {filename}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                count += 1

    print(f"\n路径修复完成! 共修改了 {count} 个文件。")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, 'docs')
    fix_image_paths(docs_dir)