#!/bin/sh
# 仓库约定（AGENTS.md「2. 目录结构与职责 / 分支职责」）的判定实现：
#   main 只能通过 merge（含 --ff-only）或 rebase 推进。
#
# 放行：HEAD 游离（rebase / cherry-pick 的中间态之外的重放场景）、非 main 分支、合并提交（MERGE_HEAD 存在）
# 拒绝：当前分支是 main 且不是合并提交（覆盖直接 commit、--amend、cherry-pick）
#
# 由 .githooks/pre-commit 与 .githooks/prepare-commit-msg 共同调用：
#   pre-commit          —— 普通提交、--amend、合并（git 对合并走 pre-merge-commit / pre-commit）
#   prepare-commit-msg  —— cherry-pick 只触发这个，不触发 pre-commit（实测）
# 客户端钩子可被 `git commit --no-verify` 绕过；真正的强制在服务端分支保护。

branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null) || exit 0

[ "$branch" = "main" ] || exit 0

if git rev-parse --quiet --verify MERGE_HEAD >/dev/null 2>&1; then
    exit 0
fi

echo "拒绝提交：main 只能通过 merge / rebase 推进（AGENTS.md「分支职责」）。" >&2
echo "请在特性分支上提交，再以 merge 或 rebase 合入 main。" >&2
echo "确需绕过（例如修复钩子本身）可用 git commit --no-verify。" >&2
exit 1
