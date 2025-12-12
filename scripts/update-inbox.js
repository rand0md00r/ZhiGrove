const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// 📂 配置路径
const DOCS_DIR = path.join(__dirname, '../docs');
const INBOX_DIR = path.join(DOCS_DIR, '00-inbox');
const INDEX_FILE = path.join(INBOX_DIR, 'index.md');

// 🚫 忽略列表
const IGNORE_FILES = ['index.md', 'triage.md', '.DS_Store'];

// 📝 定义您的固定头部内容 (用于文件丢失或重置时的恢复)
const FIXED_HEADER = `# 📥 Inbox 工作流

> KPT日志记录：
> - Keep：今天做成了什么？（哪怕是很小的事情，积累成就感）
> - Problem：遇到了什么问题？（知识库“错题本”的来源）
> - Try（尝试）：明天打算怎么解决问题？
`;

// 🛠️ 辅助函数：获取文件最后 Git 提交时间
function getGitFileDate(filePath) {
    try {
        const dateStr = execSync(`git log -1 --format=%aI "${filePath}"`, { encoding: 'utf-8' }).trim();
        return dateStr ? new Date(dateStr) : new Date();
    } catch (e) {
        return new Date();
    }
}

function updateInboxIndex() {
    if (!fs.existsSync(INBOX_DIR)) return;

    // ==========================================
    // 1. 数据收集
    // ==========================================
    let allFiles = [];
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
                        date: getGitFileDate(fullPath)
                    });
                }
            }
        });
    }
    scanDir(INBOX_DIR);

    const recentFiles = allFiles.sort((a, b) => b.date - a.date).slice(0, 5);

    const weekStats = [];
    const dirs = fs.readdirSync(INBOX_DIR);
    dirs.forEach(dir => {
        const fullPath = path.join(INBOX_DIR, dir);
        if (fs.statSync(fullPath).isDirectory()) {
            const validFiles = fs.readdirSync(fullPath).filter(f => f.endsWith('.md') && f !== 'triage.md');
            if (validFiles.length > 0) {
                weekStats.push({ name: dir, count: validFiles.length });
            }
        }
    });
    weekStats.sort((a, b) => b.name.localeCompare(a.name));

    // ==========================================
    // 2. 生成动态内容
    // ==========================================
    let dynamicContent = `\n\n## 📥 最近更新 (Latest 5)\n\n`;
    
    if (recentFiles.length > 0) {
        recentFiles.forEach(f => {
            const dateStr = f.date.toISOString().split('T')[0];
            const linkPath = f.relPath.split(path.sep).join('/');
            dynamicContent += `- [${f.name.replace('.md', '')}](./${linkPath}) <span style="opacity:0.6; font-size:0.8em; float:right">${dateStr}</span>\n`;
        });
    } else {
        dynamicContent += `*暂无文件*\n`;
    }

    dynamicContent += `\n## 📅 待归档 (Backlog)\n\n`;
    if (weekStats.length > 0) {
        dynamicContent += `| 周期目录 | 待处理文档 |\n| :--- | :---: |\n`;
        weekStats.forEach(stat => {
            dynamicContent += `| [📂 ${stat.name}](./${stat.name}/) | **${stat.count}** |\n`;
        });
    } else {
        dynamicContent += `*🎉 Inbox 已清空！*\n`;
    }

    // ==========================================
    // 3. 核心修复：带标记的替换逻辑
    // ==========================================
    // ⚠️ 必须定义这些标记，脚本才能知道去哪里替换！
    const START_MARKER = '';
    const END_MARKER = '';

    let fileContent = '';
    
    // 如果文件不存在，直接用固定头部+动态内容创建
    if (!fs.existsSync(INDEX_FILE)) {
        const initialContent = `${FIXED_HEADER}\n${START_MARKER}${dynamicContent}${END_MARKER}\n`;
        fs.writeFileSync(INDEX_FILE, initialContent);
        console.log('✅ Index created with header.');
        return;
    }

    fileContent = fs.readFileSync(INDEX_FILE, 'utf-8');
    const startIndex = fileContent.indexOf(START_MARKER);
    const endIndex = fileContent.indexOf(END_MARKER);

    if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
        // ✅ 正常情况：保留头部，替换中间
        const before = fileContent.substring(0, startIndex + START_MARKER.length);
        // 注意：这里我们保留了 before（也就是保留了您的 KPT 头部）
        
        const after = fileContent.substring(endIndex);
        
        const finalContent = before + dynamicContent + after;

        if (finalContent !== fileContent) {
            fs.writeFileSync(INDEX_FILE, finalContent);
            console.log('✅ Index updated (Header preserved).');
        } else {
            console.log('⚡ Content is up-to-date.');
        }
    } else {
        // ❌ 异常情况：找不到标记，重置整个文件
        console.warn('⚠️ Markers not found! Resetting file and restoring header...');
        // 这里会自动把您的 KPT 头部加回去
        const resetContent = `${FIXED_HEADER}\n${START_MARKER}${dynamicContent}${END_MARKER}\n`;
        fs.writeFileSync(INDEX_FILE, resetContent);
        console.log('✅ File reset with proper structure.');
    }
}

updateInboxIndex();