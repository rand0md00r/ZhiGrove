import { defineConfig } from 'vitepress'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import type { DefaultTheme } from 'vitepress'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const dirLabel: Record<string, string> = {
  '00-inbox': 'Inbox',
  '10-knowledge': 'Knowledge',
  '20-papers': 'Papers',
  '30-ideas': 'Ideas',
  '40-experiments': 'Experiments',
  '50-reports': 'Reports'
}

function mdItemsInDir(root: string, baseRoute: string): DefaultTheme.SidebarItem[] {
  try {
    return fs
      .readdirSync(root, { withFileTypes: true })
      .filter((f) => f.isFile() && f.name.endsWith('.md'))
      .filter((f) => !['readme.md', 'index.md'].includes(f.name.toLowerCase()))
      .map((f) => {
        const stem = f.name.replace(/\.md$/, '')
        const route = path.posix.join('/', baseRoute, stem)
        return { text: stem, link: route }
      })
  } catch (e) {
    return []
  }
}

function walkDir(root: string, baseRoute: string, rel = ''): DefaultTheme.SidebarItem[] {
  const abs = rel ? path.join(root, rel) : root
  if (!fs.existsSync(abs)) return []

  const mdItems = mdItemsInDir(abs, rel ? path.posix.join(baseRoute, rel) : baseRoute)

  const dirItems = fs
    .readdirSync(abs, { withFileTypes: true })
    .filter((f) => f.isDirectory())
    .map((dir) => {
      const childRel = rel ? path.join(rel, dir.name) : dir.name
      const childItems = walkDir(root, baseRoute, childRel)
      return childItems.length > 0
        ? {
            text: dir.name,
            collapsed: true,
            items: childItems
          }
        : null
    })
    .filter((v): v is DefaultTheme.SidebarItem => Boolean(v))

  return [...mdItems, ...dirItems]
}

function buildSidebar(dirName: string): DefaultTheme.SidebarItem[] {
  // 确保路径解析正确：从 docs/.vitepress 往上走一级到 docs，再进入子目录
  const root = path.resolve(__dirname, '..', dirName)
  
  if (!fs.existsSync(root)) return []

  const items = walkDir(root, dirName)
  const label = dirLabel[dirName] ?? dirName

  return [
    {
      text: label,
      items: [{ text: 'README', link: `/${dirName}/` }, ...items]
    }
  ]
}

export default defineConfig({
  title: "ZhiGrove",
  description: "Wang Yaqi's Knowledge Base",
  
  // 忽略死链，确保构建不中断
  ignoreDeadLinks: true,

  srcDir: '.', 

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '收件箱', link: '/00-inbox/' },
      { text: '知识库', link: '/10-knowledge/' },
      { text: '论文', link: '/20-papers/' },
      { text: '灵感', link: '/30-ideas/' },
      { text: '实验', link: '/40-experiments/' },
      { text: '报告', link: '/50-reports/' }
    ],

    // 修复点：全部使用 buildSidebar 函数自动生成侧边栏
    // 之前报错是因为调用了未定义的 getInboxSidebar
    sidebar: {
      '/00-inbox/': buildSidebar('00-inbox'),
      '/10-knowledge/': buildSidebar('10-knowledge'),
      '/20-papers/': buildSidebar('20-papers'),
      '/30-ideas/': buildSidebar('30-ideas'),
      '/40-experiments/': buildSidebar('40-experiments'),
      '/50-reports/': buildSidebar('50-reports')
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/wangyaqi/ZhiGrove' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025 Wang Yaqi'
    }
  }
})