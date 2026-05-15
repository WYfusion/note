#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
OVERWRITE=1
START_DIR=""

usage(){
  cat <<EOF
Usage: $0 [DIR] [-p DIR] [--dry-run] [--overwrite]

递归遍历指定目录下的所有子目录；对每个子目录 name，查找其同级的 name.md（或者 Notion 风格格式：带点的数字前缀等），
若存在则将该 md 移入对应的子目录下（不会创建缺失的同名文件夹）。默认开启 --overwrite。

Options:
  [DIR]               可以直接输入路径作为参数
  -p, --path DIR      起始目录
  -n, --dry-run       仅打印操作，不做实际移动
  -o, --overwrite     若目标已存在则默认覆盖（现已是默认行为）
  -h, --help          显示帮助
EOF
  exit 1
}

if [ $# -eq 0 ]; then
  usage
fi

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--path)
      START_DIR="$2"; shift 2 ;;
    -n|--dry-run)
      DRY_RUN=1; shift ;;
    -o|--overwrite)
      OVERWRITE=1; shift ;;
    -h|--help)
      usage; shift ;;
    *)
      if [ -z "$START_DIR" ]; then
        START_DIR="$1"; shift
      else
        echo "Unknown arg: $1" >&2; usage
      fi
      ;;
  esac
done

if [ -z "${START_DIR}" ]; then
  echo "错误：必须指定起始目录 (-p)" >&2
  usage
fi

if [ ! -d "${START_DIR}" ]; then
  echo "错误：指定路径不存在或不是目录：${START_DIR}" >&2
  exit 2
fi

# 递归遍历每个目录（包括起始目录本身）
find "${START_DIR}" -type d -print0 | while IFS= read -r -d '' dir; do
  parent=$(dirname -- "$dir")
  name=$(basename -- "$dir")
  
  # 候选1：精确匹配
  candidate1="$parent/$name.md"
  
  # 候选2：Notion导出引起的差异（顶级目录：例如文件夹 "1 xxx" -> 文件 "1. xxx.md"）
  candidate2=""
  if [[ "$name" =~ ^([0-9]+)\ (.*)$ ]]; then
    candidate2="$parent/${BASH_REMATCH[1]}. ${BASH_REMATCH[2]}.md"
  fi

  # 候选3：Notion 子层级导出差异（例如文件夹 "2 1 xxx" -> 文件 "2.1 xxx.md"）
  candidate3="${parent}/${name/ /.}.md"
  
  # 候选4：Notion 更深层级导出差异（将所有的前置数字间的空格替换为点号，目前暂时用替换第一个空格的方法基本能覆盖大部分）
  candidate4=""
  if [[ "$name" =~ ^([0-9]+)\ ([0-9]+)\ (.*)$ ]]; then
    candidate4="$parent/${BASH_REMATCH[1]}.${BASH_REMATCH[2]} ${BASH_REMATCH[3]}.md"
  fi
  
  # 寻找实际存在的文件
  candidate=""
  if [ -f "$candidate1" ]; then
    candidate="$candidate1"
  elif [ -n "$candidate2" ] && [ -f "$candidate2" ]; then
    candidate="$candidate2"
  elif [ -n "$candidate3" ] && [ -f "$candidate3" ]; then
    candidate="$candidate3"
  elif [ -n "$candidate4" ] && [ -f "$candidate4" ]; then
    candidate="$candidate4"
  fi

  if [ -n "$candidate" ]; then
    target="$dir/$(basename -- "$candidate")"

    if [ -f "$target" ] && [ "$OVERWRITE" -ne 1 ]; then
      echo "skip (target exists): $candidate -> $target"
      continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
      if [ -f "$target" ]; then
        echo "[DRY] overwrite: $candidate -> $target"
      else
        echo "[DRY] move: $candidate -> $target"
      fi
    else
      if [ -f "$target" ] && [ "$OVERWRITE" -eq 1 ]; then
        mv -f -- "$candidate" "$target"
        echo "moved (overwritten): $candidate -> $target"
      else
        mv -- "$candidate" "$target"
        echo "moved: $candidate -> $target"
      fi
    fi
  fi
done
