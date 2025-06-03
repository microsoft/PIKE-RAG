# Git 仓库配置备忘录

## 项目信息
- **项目名称**: PIKE-RAG
- **个人Fork仓库**: https://github.com/imfangs/PIKE-RAG
- **原始仓库**: https://github.com/microsoft/PIKE-RAG

## 当前Git远程仓库配置

```bash
origin  https://github.com/imfangs/PIKE-RAG (fetch)
origin  https://github.com/imfangs/PIKE-RAG (push)
upstream        https://github.com/microsoft/PIKE-RAG (fetch)
upstream        https://github.com/microsoft/PIKE-RAG (push)
```

### 配置说明
- **origin**: 指向个人fork的仓库，用于日常开发和推送
- **upstream**: 指向原始Microsoft仓库，用于同步最新更改

## 配置过程记录

### 1. 修改origin指向个人仓库
```bash
git remote set-url origin https://github.com/imfangs/PIKE-RAG
```

### 2. 添加upstream指向原始仓库
```bash
git remote add upstream https://github.com/microsoft/PIKE-RAG
```

### 3. 验证配置
```bash
git remote -v
```

## 常用Git操作

### 基本操作
```bash
# 查看仓库状态
git status

# 查看远程仓库配置
git remote -v

# 查看分支
git branch -a
```

### 推送代码到个人仓库
```bash
# 添加更改
git add .

# 提交更改
git commit -m "你的提交信息"

# 推送到个人fork仓库
git push origin main
```

### 同步原始仓库的更新
```bash
# 从上游仓库获取最新更改
git fetch upstream

# 合并上游的main分支到本地main分支
git checkout main
git merge upstream/main

# 或者直接拉取合并
git pull upstream main

# 推送合并后的代码到个人仓库
git push origin main
```

### 创建功能分支
```bash
# 创建并切换到新分支
git checkout -b feature/新功能名称

# 推送新分支到个人仓库
git push origin feature/新功能名称
```

### 处理合并冲突
```bash
# 如果出现冲突，手动解决后
git add .
git commit -m "解决合并冲突"
git push origin main
```

## 项目开发工作流

### 1. 开始新功能开发
```bash
# 确保在最新的main分支
git checkout main
git pull upstream main

# 创建功能分支
git checkout -b feature/功能名称

# 开发完成后推送
git push origin feature/功能名称
```

### 2. 创建Pull Request
- 在GitHub上从你的fork仓库向原始仓库创建PR
- 或在你的fork内部进行代码审查

### 3. 定期同步
```bash
# 定期同步原始仓库的更新
git checkout main
git pull upstream main
git push origin main
```

## Demo应用相关

### Streamlit Demo启动
```bash
# 进入demo目录
cd examples/acr_demo

# 启动应用
streamlit run app.py

# 访问地址: http://localhost:8501
```

### Demo文件结构
```
examples/acr_demo/
├── app.py              # 主应用文件
├── loggings.jsonl      # 预计算的演示数据
└── images/             # 图片资源
    ├── demo/          # 演示流程图片
    └── logo/          # Logo文件
```

## 注意事项

1. **推送前检查**: 始终在推送前检查要提交的文件
2. **分支管理**: 为不同功能创建独立分支
3. **定期同步**: 定期从upstream同步最新更改
4. **提交信息**: 使用清晰、描述性的提交信息
5. **冲突处理**: 遇到合并冲突时仔细检查和测试

## 有用的Git别名

可以将这些添加到 `~/.gitconfig` 文件中：

```ini
[alias]
    st = status
    co = checkout
    br = branch
    cm = commit -m
    ps = push
    pl = pull
    lg = log --oneline --graph --decorate --all
    sync = !git checkout main && git pull upstream main && git push origin main
```

---

*最后更新时间: 2025年6月3日* 