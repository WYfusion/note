import fs from "fs";
import path from "path";
import {
  convertMermaidToExcalidrawMarkdown,
  detectFontSizeFromMermaid,
} from "./.obsidian/1.mjs";
// 推荐执行命令（在仓库根目录运行）:
// 1) 先 dry-run 预览（推荐）
//    node mermaid.mjs --root "Notion/Qwen3-TTS Technical Report" --dry-run
// 2) 正式批量转换（最简参数，默认输出到 assets/Excalidraw）
//    node mermaid.mjs --root "Notion/Qwen3-TTS Technical Report"
// 3) 指定输出目录
//    node mermaid.mjs --root "Notion/Qwen3-TTS Technical Report" --output-dir "assets/Excalidraw" --verbose
// 4) 需要替换原 Mermaid 代码块时显式开启
//    node mermaid.mjs --root "Notion/Qwen3-TTS Technical Report" --replace-mermaid

const DEFAULT_EMBED_WIDTH = 800;
const DEFAULT_FONT_SIZE = 20;
const DEFAULT_SUBGRAPH_MODE = "auto";
const DEFAULT_OUTPUT_DIR = path.join("assets", "Excalidraw");
const SKIP_DIRS = new Set([".git", ".obsidian", "node_modules"]);

function fail(message) {
  console.error(message);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    root: "",
    embedWidth: DEFAULT_EMBED_WIDTH,
    fontSize: DEFAULT_FONT_SIZE,
    dryRun: false,
    verbose: false,
    failFast: false,
    subgraphs: DEFAULT_SUBGRAPH_MODE,
    outputDir: DEFAULT_OUTPUT_DIR,
    replaceMermaid: false,
  };

  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--root") {
      args.root = argv[++index];
    } else if (arg === "--embed-width") {
      args.embedWidth = Number(argv[++index]);
    } else if (arg === "--font-size") {
      args.fontSize = Number(argv[++index]);
    } else if (arg === "--dry-run") {
      args.dryRun = true;
    } else if (arg === "--verbose") {
      args.verbose = true;
    } else if (arg === "--fail-fast") {
      args.failFast = true;
    } else if (arg === "--subgraphs") {
      args.subgraphs = argv[++index];
    } else if (arg === "--output-dir") {
      args.outputDir = argv[++index];
    } else if (arg === "--replace-mermaid") {
      args.replaceMermaid = true;
    } else if (arg === "--help") {
      printHelp();
      process.exit(0);
    } else {
      fail(`Unknown argument: ${arg}`);
    }
  }

  if (!args.root) {
    fail("Missing required argument: --root");
  }
  if (!["auto", "always", "never"].includes(args.subgraphs)) {
    fail("Invalid value for --subgraphs. Use auto, always, or never.");
  }
  if (!Number.isFinite(args.embedWidth) || args.embedWidth <= 0) {
    fail("Invalid value for --embed-width");
  }
  if (!Number.isFinite(args.fontSize) || args.fontSize <= 0) {
    fail("Invalid value for --font-size");
  }
  if (!args.outputDir || !String(args.outputDir).trim()) {
    fail("Invalid value for --output-dir");
  }

  return args;
}

function printHelp() {
  console.log(`Usage:
  node mermaid.mjs --root <folder> [options]

Options:
  --embed-width <number>   Obsidian embed width, default 800
  --font-size <number>     Fallback font size when Mermaid does not specify one
  --subgraphs <mode>       auto | always | never, default auto
  --output-dir <path>      Output directory, default assets/Excalidraw
  --replace-mermaid        Replace Mermaid code blocks with Excalidraw embeds
  --dry-run                Preview changes without writing files
  --verbose                Print each converted block
  --fail-fast              Stop immediately on the first failed block

Behavior:
  - Recursively scans all .md files under --root
  - Skips .excalidraw.md files
  - Converts each Mermaid fenced block into a native Excalidraw file under --output-dir
  - By default keeps original Mermaid blocks unchanged
  - When --replace-mermaid is set, replaces Mermaid blocks with ![[...|800]] embeds
  - Avoids overwriting existing .excalidraw.md files by allocating a unique name globally`);
}

function collectMarkdownFiles(rootDir) {
  const files = [];

  function walk(currentDir) {
    for (const entry of fs.readdirSync(currentDir, { withFileTypes: true })) {
      if (entry.name.startsWith(".") && entry.name !== ".") {
        if (entry.isDirectory()) {
          continue;
        }
      }
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) {
          walk(path.join(currentDir, entry.name));
        }
        continue;
      }
      if (!entry.name.endsWith(".md")) {
        continue;
      }
      if (entry.name.endsWith(".excalidraw.md")) {
        continue;
      }
      files.push(path.join(currentDir, entry.name));
    }
  }

  walk(rootDir);
  return files;
}

function findMermaidBlocks(content) {
  const pattern = /```mermaid[^\n\r]*\r?\n([\s\S]*?)```/g;
  const blocks = [];
  let match;

  while ((match = pattern.exec(content)) !== null) {
    blocks.push({
      start: match.index,
      end: pattern.lastIndex,
      source: match[1].trim(),
    });
  }

  return blocks;
}

function parseHeadings(content) {
  const headings = [];
  const pattern = /^(#{1,6})\s+(.+)$/gm;
  let match;

  while ((match = pattern.exec(content)) !== null) {
    headings.push({
      title: match[2].trim(),
      start: match.index,
    });
  }

  return headings;
}

function nearestHeading(headings, position) {
  let current = null;
  for (const heading of headings) {
    if (heading.start >= position) {
      break;
    }
    current = heading;
  }
  return current;
}

function sanitizeFileName(name) {
  const sanitized = String(name || "")
    .replace(/[\\/:*?"<>|]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[. ]+$/g, "");
  return sanitized || "Mermaid 图";
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function buildDiagramBaseName(notePath, headingTitle, blockIndex, totalBlocks) {
  const noteStem = sanitizeFileName(path.basename(notePath, ".md"));
  const heading = headingTitle ? sanitizeFileName(headingTitle) : "";

  if (heading && heading !== noteStem) {
    if (totalBlocks === 1) {
      return `${noteStem} - ${heading}`;
    }
    return `${noteStem} - ${heading} - 图 ${pad2(blockIndex + 1)}`;
  }

  if (totalBlocks === 1) {
    return noteStem;
  }

  return `${noteStem} - 图 ${pad2(blockIndex + 1)}`;
}

function collectExistingOutputNames(outputDir) {
  const usedNames = new Set();
  if (!fs.existsSync(outputDir)) {
    return usedNames;
  }
  for (const entry of fs.readdirSync(outputDir, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith(".excalidraw.md")) {
      continue;
    }
    const baseName = path.basename(entry.name, ".excalidraw.md").toLowerCase();
    usedNames.add(baseName);
  }
  return usedNames;
}

function allocateDiagramFile(baseName, outputDir, usedNames) {
  let candidateBase = baseName;
  let counter = 2;
  let candidatePath = path.join(outputDir, `${candidateBase}.excalidraw.md`);

  while (
    usedNames.has(candidateBase.toLowerCase()) ||
    fs.existsSync(candidatePath)
  ) {
    candidateBase = `${baseName} (${counter})`;
    candidatePath = path.join(outputDir, `${candidateBase}.excalidraw.md`);
    counter += 1;
  }

  usedNames.add(candidateBase.toLowerCase());

  const fileName = path.basename(candidatePath);
  return {
    outputPath: candidatePath,
    embedTarget: fileName.replace(/\.md$/i, ""),
    fileName,
  };
}

function replacementEmbed(embedTarget, width) {
  return `![[${embedTarget}|${width}]]`;
}

function replaceBlocks(content, replacements) {
  let cursor = 0;
  let result = "";

  for (const replacement of replacements) {
    result += content.slice(cursor, replacement.start);
    result += replacement.value;
    cursor = replacement.end;
  }

  result += content.slice(cursor);
  return result;
}

async function convertBlock({
  notePath,
  block,
  blockIndex,
  totalBlocks,
  heading,
  args,
  usedNames,
}) {
  const baseName = buildDiagramBaseName(
    notePath,
    heading?.title || "",
    blockIndex,
    totalBlocks,
  );
  const { outputPath, embedTarget, fileName } = allocateDiagramFile(
    baseName,
    args.outputDir,
    usedNames,
  );
  const fontSize = detectFontSizeFromMermaid(block.source, args.fontSize);
  const shouldFlattenAlways = args.subgraphs === "always";
  const shouldAutoFlatten = args.subgraphs === "auto";

  const { markdown, flattened } = await convertMermaidToExcalidrawMarkdown({
    mermaidSource: block.source,
    fontSize,
    flattenSubgraphs: shouldFlattenAlways,
    autoFlattenSubgraphs: shouldAutoFlatten,
  });

  if (!args.dryRun) {
    fs.writeFileSync(outputPath, markdown, "utf8");
  }

  return {
    fileName,
    embed: replacementEmbed(embedTarget, args.embedWidth),
    flattened,
  };
}

async function processNote(notePath, args, usedNames) {
  const content = fs.readFileSync(notePath, "utf8");
  const blocks = findMermaidBlocks(content);
  if (blocks.length === 0) {
    return {
      notePath,
      scanned: 0,
      converted: 0,
      changed: false,
      failures: [],
    };
  }

  const headings = parseHeadings(content);
  const replacements = [];
  const failures = [];
  let convertedCount = 0;

  for (let index = 0; index < blocks.length; index += 1) {
    const block = blocks[index];
    const heading = nearestHeading(headings, block.start);

    try {
      const result = await convertBlock({
        notePath,
        block,
        blockIndex: index,
        totalBlocks: blocks.length,
        heading,
        args,
        usedNames,
      });

      convertedCount += 1;
      if (args.replaceMermaid) {
        replacements.push({
          start: block.start,
          end: block.end,
          value: result.embed,
        });
      }

      if (args.verbose) {
        console.log(
          `[converted] ${path.relative(process.cwd(), notePath)} -> ${result.fileName}${
            result.flattened ? " (flattened subgraphs)" : ""
          }`,
        );
      }
    } catch (error) {
      failures.push({
        notePath,
        blockIndex: index,
        heading: heading?.title || "",
        error: error?.message || String(error),
      });
      if (args.verbose || args.failFast) {
        console.error(
          `[failed] ${path.relative(process.cwd(), notePath)} block ${index + 1}: ${error?.message || error}`,
        );
      }
      if (args.failFast) {
        throw error;
      }
    }
  }

  if (!args.replaceMermaid) {
    return {
      notePath,
      scanned: blocks.length,
      converted: convertedCount,
      changed: false,
      failures,
    };
  }

  if (replacements.length === 0) {
    return {
      notePath,
      scanned: blocks.length,
      converted: convertedCount,
      changed: false,
      failures,
    };
  }

  const updatedContent = replaceBlocks(content, replacements);
  if (!args.dryRun) {
    fs.writeFileSync(notePath, updatedContent, "utf8");
  }

  return {
    notePath,
    scanned: blocks.length,
    converted: convertedCount,
    changed: updatedContent !== content,
    failures,
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const rootPath = path.resolve(args.root);
  if (!fs.existsSync(rootPath) || !fs.statSync(rootPath).isDirectory()) {
    fail(`Root folder does not exist: ${rootPath}`);
  }
  args.outputDir = path.resolve(args.outputDir);
  if (!args.dryRun) {
    fs.mkdirSync(args.outputDir, { recursive: true });
  }
  const usedNames = collectExistingOutputNames(args.outputDir);

  const notes = collectMarkdownFiles(rootPath);
  const summary = {
    scannedFiles: notes.length,
    touchedFiles: 0,
    scannedBlocks: 0,
    convertedBlocks: 0,
    failures: [],
  };

  for (const notePath of notes) {
    const result = await processNote(notePath, args, usedNames);
    summary.scannedBlocks += result.scanned;
    summary.convertedBlocks += result.converted;
    if (result.changed) {
      summary.touchedFiles += 1;
    }
    summary.failures.push(...result.failures);
  }

  console.log(
    JSON.stringify(
      {
        root: rootPath,
        outputDir: args.outputDir,
        dryRun: args.dryRun,
        scannedFiles: summary.scannedFiles,
        touchedFiles: summary.touchedFiles,
        scannedBlocks: summary.scannedBlocks,
        convertedBlocks: summary.convertedBlocks,
        failedBlocks: summary.failures.length,
        failures: summary.failures,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
