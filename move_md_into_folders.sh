#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
OVERWRITE=0
START_DIR=""

usage(){
  cat <<EOF
Usage: $0 -p DIR [--dry-run] [--overwrite]

递归遍历指定目录下的所有子目录；对每个子目录 name，查找其同级的 name.md，
若存在则将该 md 移入对应的子目录下（不会创建缺失的同名文件夹）。

Options:
  -p, --path DIR      起始目录（必需）
  -n, --dry-run       仅打印操作，不做实际移动
  -o, --overwrite     若目标已存在则覆盖
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
      echo "Unknown arg: $1" >&2; usage ;;
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
  candidate="$parent/$name.md"

  if [ -f "$candidate" ]; then
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
