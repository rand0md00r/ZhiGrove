// scripts/update-inbox.js
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// 📂 配置路径
const DOCS_DIR = path.join(__dirname, '../docs');
const INBOX_DIR = path.join(DOCS_DIR, '00-inbox');
const INDEX_FILE = path.join(INBOX_DIR, 'index.md');

// 🚫 忽略列表
const IGNORE_FILES = ['index.md', 'triage.md', '.DS_Store'];

// 🛠️ 辅助函数：获取文件最后 Git 提交时间
function getGitFileDate(filePath) {
    try {
        // 使用 git log 获取 ISO 格式的时间
        const dateStr = execSync(`git log -1 --format=%aI "${filePath}"`, { encoding: 'utf-8' }).trim();
        return dateStr ? new Date(dateStr) : new Date(); // 如果是新文件未提交，回退到当前时间
    } catch (e) {
        return new Date();
    }
}

function updateInboxIndex() {
    if (!fs.existsSync(INBOX_DIR)) {
        console.log('Inbox dir not found, skipping.');
        return;
    }

    // ==========================================
    // 任务 1: 获取 Inbox 下所有 MD 文件并按 Git 时间排序 (最新的 5 个)
    // ==========================================
    let allFiles = [];

    // 递归扫描函数
    function scanDir(dir) {
        const files = fs.readdirSync(dir);
        files.forEach(file => {
            const fullPath = path.join(dir, file);
            if (fs.statSync(fullPath).isDirectory()) {
                scanDir(fullPath);
            } else {
                if (file.endsWith('.md') && !IGNORE_FILES.includes(file)) {
                    allFiles.push({
                        path: fullPath,
                        name: file,
                        relPath: path.relative(INBOX_DIR, fullPath),
                        // 关键：在 CI 中必须用 Git 时间
                        date: getGitFileDate(fullPath)
                    });
                }
            }
        });
    }

    scanDir(INBOX_DIR);

    // 排序并取前 5
    const recentFiles = allFiles
        .sort((a, b) => b.date - a.date)
        .slice(0, 5);

    // ==========================================
    // 任务 2: 统计 Week 目录下的未归档文件
    // ==========================================
    const weekStats = [];
    const dirs = fs.readdirSync(INBOX_DIR);

    dirs.forEach(dir => {
        const fullPath = path.join(INBOX_DIR, dir);
        if (fs.statSync(fullPath).isDirectory()) {
            // 读取该子目录下的有效 MD 文件 (排除 triage.md)
            const validFiles = fs.readdirSync(fullPath).filter(f => 
                f.endsWith('.md') && f !== 'triage.md'
            );

            if (validFiles.length > 0) {
                weekStats.push({
                    name: dir,
                    count: validFiles.length
                });
            }
        }
    });

    // 按目录名排序（通常 Week 目录是按时间命名的，倒序排列）
    weekStats.sort((a, b) => b.name.localeCompare(a.name));

    // ==========================================
    // 任务 3: 生成 Markdown 内容
    // ==========================================
    let mdContent = `\n\n## 📥 最近更新 (Latest 5)\n\n`;
    
    if (recentFiles.length > 0) {
        recentFiles.forEach(f => {
            // 格式化日期 YYYY-MM-DD
            const dateStr = f.date.toISOString().split('T')[0];
            // 替换反斜杠以适配 Windows/Linux 路径差异
            const linkPath = f.relPath.split(path.sep).join('/');
            mdContent += `- [${f.name.replace('.md', '')}](./${linkPath}) <span style="opacity:0.6; font-size:0.8em; float:right">${dateStr}</span>\n`;
        });
    } else {
        mdContent += `*暂无文件*\n`;
    }

    mdContent += `\n## 📅 待归档 (Backlog)\n\n`;
    if (weekStats.length > 0) {
        mdContent += `| 周期目录 | 待处理文档 |\n| :--- | :---: |\n`;
        weekStats.forEach(stat => {
            mdContent += `| [📂 ${stat.name}](./${stat.name}/) | **${stat.count}** |\n`;
        });
    } else {
        mdContent += `*🎉 Inbox 已清空！*\n`;
    }

    // ==========================================
    // 任务 4: 写入文件 (保留头部手动内容)
    // ==========================================
    let fileContent = '';
    if (fs.existsSync(INDEX_FILE)) {
        fileContent = fs.readFileSync(INDEX_FILE, 'utf-8');
    } else {
        fileContent = `# Inbox\n\n\n\n`;
    }

    const startMarker = '';
    const endMarker = '';
    
    const regex = new RegExp(`${startMarker}[\\s\\S]*?${endMarker}`);
    
    if (regex.test(fileContent)) {
        const newContent = fileContent.replace(regex, `${startMarker}${mdContent}${endMarker}`);
        // 只有内容真的变了才写入，避免无效 commit
        if (newContent !== fileContent) {
            fs.writeFileSync(INDEX_FILE, newContent);
            console.log('✅ Index updated.');
        } else {
            console.log('⚡ No changes needed.');
        }
    } else {
        console.log('⚠️ Markers not found in index.md');
    }
}

updateInboxIndex();