# 项目说明

## 工作区定位

这个工作区主要是一个轻量级的 Codex 测试环境，用来：

- 验证 Git、提交和推送相关行为
- 测试 MATLAB MCP Core Server 的安装、配置和连通性
- 集中放置相关二进制文件、日志和临时测试产物

## 重要文件

- `README.md`：仓库的简短说明
- `matlab-mcp-core-server-win64.exe`：MATLAB MCP Core Server 可执行文件
- `matlab-mcp-core-server/`：官方仓库的本地克隆，用于参考 README 和源码
- `matlab-mcp-logs/`：MATLAB MCP 测试过程中生成的日志目录

## 工作规则

- 优先做最小、直接、可验证的修改
- 没有明确要求时，不要删除 MATLAB 二进制文件或日志目录
- 把这个仓库视为测试和验证工作区，而不是正式产品代码库
- 测试 Git 行为时，优先使用非破坏性命令
- 测试 MATLAB MCP 时，除非用户明确要求其他模式，否则默认保持：
- `--matlab-display-mode=desktop`
- `--matlab-session-mode=new`

## 已知背景

- 本机 MATLAB 安装路径是 `D:\Program Files\MATLAB\R2025a`
- 这个仓库已经被用于验证 Codex MCP、Git push 和 Chrome 控制链路
