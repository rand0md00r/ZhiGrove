1. 下载并安装Rustdesk客户端：
    - [download](https://github.com/rustdesk/rustdesk/releases/tag/1.4.4)

2. 配置中继服务器
    - Rustdesk客户端中：点击右上角的"三"进入设置，点击“网络”，点击“ID/中继服务器“，输入以下内容：
        - ID服务器：118.196.11.76
        - 中继服务器空置；
        - API服务器空置；
        - Key中输入“7whLYzxd2yXYG9VgoNqPVYSFx7pihuoq5nnt1yCjKG8=”

3. 远程控制5090
    - 493 287 946
    - Jack@123



````markdown
# 用 RustDesk 自建远程桌面服务：从 0 搭建到流畅使用

这篇博客记录一下我自己搭建 RustDesk 远程桌面服务的全过程：  
后端用一台公网 Ubuntu 服务器跑 RustDesk Server（hbbs/hbbr），前端用 Windows + mac 客户端访问。  
最后顺手也把画面很卡、FPS 很低的调优过程整理了一下。

---

## 1. 场景与目标

- **目标**：不依赖官方服务器，完全使用自己的公网服务器作为 RustDesk 中转 / 信令，实现：
  - 在任何地方远程控制家里的 Windows / 办公电脑；
  - 连接稳定、延迟可接受，画面尽量流畅。
- **服务器环境**：
  - 公网机器：Ubuntu 24.04
  - 已安装 Docker
  - 使用镜像：`rustdesk/rustdesk-server:1.1.14`（或 `:latest`）

---

## 2. RustDesk 自建服务架构小抄

RustDesk 自建服务主要有两个进程：

- `hbbs`：信令/ID 服务器  
  - 负责客户端注册 ID、NAT 测试、打洞等。
- `hbbr`：中继服务器  
  - 打洞失败时，通过中继转发画面流。

**推荐端口（开源版）**：

- hbbs（信令）：
  - `21115/TCP`：NAT 类型测试
  - `21116/TCP + UDP`：ID 注册、心跳、打洞、连接
  - `21118/TCP`：Web 客户端（可选）
- hbbr（中继）：
  - `21117/TCP`：中继数据
  - `21119/TCP`：Web 中继（可选）

---

## 3. 服务器端搭建（Ubuntu + Docker）

### 3.1 创建数据目录

数据目录用于保存密钥、公钥和数据库（ID 信息）：

```bash
sudo mkdir -p /opt/rustdesk/data
````

### 3.2 启动 hbbs（ID 服务器）

```bash
sudo docker run --name hbbs \
  -v /opt/rustdesk/data:/root \
  -td --net=host \
  --restart unless-stopped \
  rustdesk/rustdesk-server:1.1.14 hbbs
```

关键点：

* `--net=host`：让容器直接使用宿主机网络，不用再做端口映射。
* `-v /opt/rustdesk/data:/root`：把密钥等数据持久化到宿主机。

### 3.3 启动 hbbr（中继服务器）

```bash
sudo docker run --name hbbr \
  -v /opt/rustdesk/data:/root \
  -td --net=host \
  --restart unless-stopped \
  rustdesk/rustdesk-server:1.1.14 hbbr
```

### 3.4 检查容器 + 端口监听

```bash
# 容器是否在运行
sudo docker ps

# 端口监听情况
sudo ss -lntup | grep 2111
```

理想情况下应看到类似输出：

```text
LISTEN ... 0.0.0.0:21115 ... hbbs
LISTEN ... 0.0.0.0:21116 ... hbbs
LISTEN ... 0.0.0.0:21117 ... hbbr
LISTEN ... 0.0.0.0:21118 ... hbbs
LISTEN ... 0.0.0.0:21119 ... hbbr
```

说明服务在宿主机上成功监听端口。

---

## 4. 从日志中获取 Key（公钥）

RustDesk 客户端连接自建服务器需要一个 `Key`（服务器公钥）。

### 4.1 从日志里直接查看

```bash
sudo docker logs hbbs | grep -i "Key"
```

可以看到类似：

```text
INFO ... Key: 7whLYzxd2yXYG9VgoNqPVYSFx7pihuoq5nnt1yCjKG8=
```

> `=` 之前那一整串就是要填到客户端里的 Key。

### 4.2 从公钥文件读取

公钥文件默认生成在数据目录：

```bash
sudo cat /opt/rustdesk/data/id_ed25519.pub
```

输出示例：

```text
7whLYzxd2yXYG9VgoNqPVYSFx7pihuoq5nnt1yCjKG8=
```

复制 **整行内容**，不要带多余的 shell 提示符。

---

## 5. 云服务器安全组配置与验证

很多连接失败都卡在**安全组没放行端口**。

### 5.1 入站规则建议

对于绑定公网 IP 的那块网卡，对应安全组的 **入向规则** 建议：

| 协议  | 端口范围        | 源地址       | 说明          |
| --- | ----------- | --------- | ----------- |
| TCP | 21115-21119 | 0.0.0.0/0 | hbbs + hbbr |
| UDP | 21116       | 0.0.0.0/0 | hbbs 打洞     |
| TCP | 22          | 0.0.0.0/0 | SSH（已有即可）   |

常见坑：有的云厂商默认规则写成“源地址：本安全组”，这只允许安全组**内部机器互访**，外部网络（家里、办公室）是访问不了的，要改成 `0.0.0.0/0` 或指定 IP 段。

### 5.2 出站规则

一般云厂商默认是全部放行，可以不用管。
如果有限制，也要确保：

* `21115-21119/TCP`、`21116/UDP` 对外网可出；
* 或者直接 `ALL / 0.0.0.0/0`。

### 5.3 用 nc 从外网测试端口

在本地电脑（例如 mac）上：

```bash
# 测试 ssh，确认 IP 真的通
nc -vz YOUR_SERVER_IP 22

# 测试 hbbs / hbbr 端口
nc -vz YOUR_SERVER_IP 21116
nc -vz YOUR_SERVER_IP 21117
```

* 如果 22 通、21116/21117 不通，多半是安全组没放开；
* 修改安全组后，再测一遍直到 `succeeded` 为止。

---

## 6. 客户端配置（Windows / mac）

### 6.1 安装 RustDesk 客户端

去 RustDesk 官网或 GitHub Releases 下载对应平台安装包：

* Windows：`.exe`
* macOS：`.dmg`

安装完成后启动即可。

### 6.2 配置“网络 / Network”

1. 打开 RustDesk，点击左侧 **设置 / Settings**。

2. 进入 **网络 / Network**。

3. 点击右上角的小锁图标，授予管理员权限（Windows 可能会弹 UAC）。

4. 填写自建服务器信息：

   * **ID 服务器 / ID Server**：

     ```text
     YOUR_SERVER_IP
     ```

     > 不写端口时默认用 21116。

   * **中继服务器 / Relay Server**：

     可以先留空，让客户端自动从 ID 服务器获取；
     或者写死：

     ```text
     YOUR_SERVER_IP:21117
     ```

   * **Key**：

     填前面拿到的那串公钥，例如：

     ```text
     7whLYzxd2yXYG9VgoNqPVYSFx7pihuoq5nnt1yCjKG8=
     ```

5. 保存后关闭再重启 RustDesk，等待几秒，看本机的 ID 是否正常出现。

### 6.3 建立远程连接

* 在被控端（例如公司 Windows）上：

  * 打开 RustDesk，保持在线。
* 在控制端（自己的 mac）上：

  * 在“连接到伙伴 / ID”框里输入目标机器的 RustDesk ID；
  * 点击连接，输入对方密码或确认，即可远控。

---

## 7. 画面卡顿 / FPS 低的优化方案

连接成功之后，默认画面质量不一定理想，尤其是走中继时可能只有个位数 FPS。下面是几条实测有效的优化建议。

### 7.1 会话内设置：调整图像质量与 FPS

在已经连上的远控窗口顶部工具栏中：

1. 点击显示设置图标（小显示器）。
2. 把“图像质量 / Image quality”改成 **自定义 / Custom**。
3. 调整滑块：

   * **Quality**：建议 70–90；
   * **FPS**：拉到 **30** 或 **60**。

这一步对流畅度影响非常大。

### 7.2 客户端全局设置：默认显示参数

为了每次连接都默认高 FPS，可以在客户端全局设置：

1. 打开 RustDesk → **设置 / Settings → 显示 / Display**。
2. 设置：

   * 默认图像质量：`Custom`
   * 自定义图像质量：`50–80`
   * 自定义 FPS：`30` 或 `60`
   * 默认编码器：先 `Auto`，不满意可以尝试 `AV1` 或 `H264` / `VP9`。

部分显卡 / CPU 对 H265 的硬件编码支持一般，换编码器有时能从 6–10 FPS 拉到 20+。

### 7.3 分辨率与缩放

如果远端是 2K/4K 显示器，编码压力和带宽需求会明显提高。

建议：

* 在远端系统里把分辨率调到 **1920×1080**；
* 或者在 RustDesk 的显示菜单里设置缩放为 80%、70%，减少传输数据量。

### 7.4 Relay vs Direct（中继 vs 直连）

状态栏中有时会显示当前连接类型：

* `Direct` / `P2P`：
  两端直接互联，延迟和带宽都更好。
* `Relay`：
  所有数据通过你的服务器转发，受服务器上下行带宽限制。

如果两端本身都是家宽 / 宽带环境，而服务器是轻量云上行较低，走 relay 时的 FPS 会明显掉。
改进思路：

* 尽量保证服务器有足够带宽（上行至少 10 Mbps 以上更稳）；
* 或者给两端搭一个 VPN（如 Tailscale），让 RustDesk 在 VPN 地址上直连。

### 7.5 其他细节检查

* **避免 CPU/GPU 满载**
  打开任务管理器/活动监视器，看 RustDesk 的 CPU/GPU 占用是否爆表。
* **关闭隐私模式测试**
  某些版本开启隐私模式时会显著降低帧率，遇到奇怪卡顿可以暂时关闭试试。

---

## 8. 常见问题排查清单

最后给一个“踩坑 checklist”，遇到问题可以按顺序对照：

1. **容器是否在运行？**

   ```bash
   sudo docker ps
   ```

2. **端口是否监听？**

   ```bash
   sudo ss -lntup | grep 2111
   ```

3. **安全组是否放行公网 IP？**

   * 入向规则中是否存在 `TCP 21115-21119`、`UDP 21116`，源地址为 `0.0.0.0/0`？

4. **从外网 `nc` 是否能连上？**

   ```bash
   nc -vz YOUR_SERVER_IP 21116
   nc -vz YOUR_SERVER_IP 21117
   ```

5. **客户端 Network 设置是否正确？**

   * ID Server：`YOUR_SERVER_IP`
   * Relay：`YOUR_SERVER_IP:21117` 或留空
   * Key：与服务器 `id_ed25519.pub` 完全一致

6. **画面卡顿时：**

   * 会话内设置是否切到 `Custom`，FPS 设为 30/60？
   * 是否为高分辨率（2K/4K），是否尝试降低分辨率？
   * 连接类型是 `Direct` 还是 `Relay`？

---

## 9. 小结

整套自建流程其实就三步：

1. **服务器端**：Docker 起 `hbbs` + `hbbr`，确认端口监听；
2. **云厂商安全组**：把 21115–21119 TCP 和 21116 UDP 对公网放开；
3. **客户端**：在 Network 中填好 `ID Server` / `Relay` / `Key`，在 Display 中调高 FPS 和画质。

搭好之后，RustDesk 基本就是“自己的 AnyDesk/TeamViewer”，
既能穿透内网，又不依赖第三方服务器，适合长期远程维护自己的机器。

如果后续要做多用户共享、Web 客户端访问之类的，也可以在这个基础上继续扩展。

```
::contentReference[oaicite:0]{index=0}
```
