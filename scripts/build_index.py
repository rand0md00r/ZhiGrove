#!/usr/bin/env python3
import glob, os, re
from datetime import datetime

def latest(md_glob, n=5):
    files = sorted(glob.glob(md_glob, recursive=True), key=os.path.getmtime, reverse=True)
    out = []
    for f in files[:n]:
        name = os.path.splitext(os.path.basename(f))[0]
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d")
        out.append(f"- {mtime} · [{name}]({f})")
    return "\n".join(out)

readme = open("README.md", "r", encoding="utf-8").read()
block = f"""
### 近期更新

**论文**  
{latest('20-papers/**/*.md')}

**闪念/会议**  
{latest('30-ideas/**/*.md')}

**实验**  
{latest('40-experiments/**/log.md')}
""".strip()

new = re.sub(r"### 近期更新(.|\n)*?(?=^#|\Z)", block + "\n\n", readme, flags=re.MULTILINE)
open("README.md", "w", encoding="utf-8").write(new)
print("README updated.")
