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
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((f) => f.isFile() && f.name.endsWith('.md'))
    .filter((f) => !['readme.md', 'index.md'].includes(f.name.toLowerCase()))
    .map((f) => {
      const stem = f.name.replace(/\.md$/, '')
      const route = path.posix.join('/', baseRoute, stem)
      return { text: stem, link: route }
    })
}

function walkDir(root: string, baseRoute: string, rel = ''): DefaultTheme.SidebarItem[] {
  const abs = rel ? path.join(root, rel) : root
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
  
  // 指向 docs 目录，因为你的 markdown 都在这里
  srcDir: '.', 

  themeConfig: {
    // 网站左上角的 Logo（可选）
    // logo: '/assets/logo.png',

    // 顶部导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '收件箱 (Inbox)', link: '/00-inbox/' },
      { text: '知识库 (Knowledge)', link: '/10-knowledge/' },
      { text: '论文 (Papers)', link: '/20-papers/' },
      { text: '灵感 (Ideas)', link: '/30-ideas/' },
      { text: '实验 (Experiments)', link: '/40-experiments/' },
      { text: '报告 (Reports)', link: '/50-reports/' }
    ],

    // 侧边栏配置 - 这里配置不同目录下的侧边栏显示
    sidebar: {
      '/00-inbox/': buildSidebar('00-inbox'),
      '/10-knowledge/': buildSidebar('10-knowledge'),
      '/20-papers/': buildSidebar('20-papers'),
      '/30-ideas/': buildSidebar('30-ideas'),
      '/40-experiments/': buildSidebar('40-experiments'),
      '/50-reports/': buildSidebar('50-reports')
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/你的github用户名/ZhiGrove' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025 Wang Yaqi'
    }
  },

  // 图片资源重定向规则（确保 markdown 中的 /assets/ 能被正确解析）
  rewrites: {
    // 如果有特殊的路径映射可以在这里配置，目前标准配置即可
  }
})
