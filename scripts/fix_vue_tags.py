import os
import re

def fix_files(root_dir):
    print(f"开始修复目录: {root_dir} 中的 Vue 语法冲突...")
    count = 0
    
    # 允许的合法 HTML 标签白名单（这些不会被转义）
    allowed_tags = {
        'div', 'span', 'p', 'br', 'hr', 'img', 'a', 'b', 'i', 'strong', 'em', 
        'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'thead', 'tbody', 
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre', 'code',
        'details', 'summary', 'iframe', 'video', 'audio', 'source'
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.md'):
                continue
                
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # ------------------------------------------------------------------
            # 核心逻辑：使用回调函数处理每一个 <...>
            # ------------------------------------------------------------------
            def replace_tag(match):
                full_match = match.group(0) # 例如 <unk> 或 <div class="...">
                tag_name = match.group(1).lower() # 例如 unk 或 div

                # 如果是 HTML 注释，或者在白名单里，或者以 / 开头（结束标签），则保留原样
                if full_match.startswith('<!--') or tag_name in allowed_tags or tag_name.startswith('/'):
                    return full_match
                
                # 否则，认为是非法标签，进行转义：<unk> -> &lt;unk>
                # 这样页面上依然显示 <unk>，但不会被当做组件编译
                return full_match.replace('<', '&lt;')

            # 正则解释：
            # (?<!`) : 前面不能有反引号 (避免修改代码块内的内容，简单处理)
            # <([a-zA-Z][a-zA-Z0-9_\-]*) : 匹配 <开头，后跟标签名
            # [^>]*> : 匹配剩余属性直到 >
            
            # 注意：这个正则处理简单的行内代码块保护（通过逐行处理会更稳健，但全文替换效率高）
            # 为了防止误伤代码块，我们只替换那些明显像标签但不在白名单里的
            
            # 这里使用一个简化的策略：只处理正文中明显错误的标签
            # 复杂的代码块保护逻辑比较难通过简单正则实现，
            # 但通常代码块里的 <unk> 是被 ``` 包裹的，不会触发 VitePress 错误（除非是高亮语言错误）
            # 只有正文中的 <unk> 会报错。
            
            new_content = re.sub(r'<([a-zA-Z][a-zA-Z0-9_\-]*)([^>]*)>', replace_tag, content)

            # 额外的：处理孤立的 < 符号 (如 x < y)，如果后面跟空格通常没事，但紧跟字母会报错
            # new_content = re.sub(r'(?<!&lt;) <(?=[a-zA-Z])', ' &lt;', new_content)

            if new_content != content:
                print(f"🛠️  修复: {filename}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                count += 1

    print(f"\n修复完成! 共修改了 {count} 个文件。")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, 'docs')
    fix_files(docs_dir)