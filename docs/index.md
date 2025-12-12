# ZhiGrove · 知林

## 知识库工具脚本
```bash
# 迁移所有 Markdown 中的本地图片到 /assets，并改写链接
find docs -name "*.md" -print0 | xargs -0 -n1 python scripts/relocate_images.py

# 清理未被引用的图片（先预览，再执行）
python scripts/clean_unused_assets.py --dry-run
python scripts/clean_unused_assets.py
```
