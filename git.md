# Nematic Git 参考手册

当前项目路径: `/home/wangshengping/04_Nero/code/Nematic`

## 1. 连接服务器与配置 Git

**通过 SSH 连接服务器进入项目:**
```bash
ssh wangshengping@10.152.88.66 "cd /home/wangshengping/04_Nero/code/Nematic && pwd && ls"
```

**确保后台 SSH 密钥已启动 (每次重启终端可能需要):**
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## 2. 日常更新与推送 (Push)

每次修改完代码（如 `train.py` 或 `networks` 目录），执行以下三步将代码推送到云端：

```bash
git add .
git commit -m "更新了 train.py 和网络结构"
git push origin main
```

## 3. 拉取远端更新 (Pull)

将 GitHub 上的最新修改同步至本地：

```bash
git pull origin main
```

## 4. 彻底重置仓库 (误加几 GB 文件的紧急补救)

如果你不小心执行了 `git add .` 导致巨大的 `.pth` 被打包进了缓存记录且推不上云端，使用下面指令立刻推倒重建历史（不影响本地真实文件）：

```bash
# 1. 删除被污染的历史并重构
rm -rf .git
git init
git branch -M main
git remote add origin git@github.com:SenpeWang/Nematic.git

# 2. 重新配置大文件黑名单
echo "Outputs/" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pth" >> .gitignore
echo ".codex_backups/" >> .gitignore
echo "tmux_logs" >> .gitignore


# 3. 精确添加源码
git add configs datasets mamba_ssm networks utils test.py train.py .gitignore

# 4. 强制推送以覆盖垃圾记录
git commit -m "feat: init Nematic core framework with local mamba_ssm"
git push -u origin main -f
```

## 5. 新增大文件或目录时更新黑名单

如果项目中出现了一个必定很大（如存储权重的）新目录，例如 `Weights/`，提前封锁它：

```bash
echo "Weights/" >> .gitignore
```
