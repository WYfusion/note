// node .\br.mjs --root "assets\Excalidraw"
import fs from "fs";
import path from "path";

const SKIP_DIRS = new Set([".git", ".obsidian", "node_modules"]);

function fail(message) {
  console.error(message);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    root: "",
    dryRun: false,
    verbose: false,
    failFast: false,
  };

  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--root") {
      args.root = argv[++index];
    } else if (arg === "--dry-run") {
      args.dryRun = true;
    } else if (arg === "--verbose") {
      args.verbose = true;
    } else if (arg === "--fail-fast") {
      args.failFast = true;
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

  return args;
}

function printHelp() {
  console.log(`Usage:
  node excalidraw.mjs --root <folder> [options]

Options:
  --dry-run     Preview changes without writing files
  --verbose     Print each changed file
  --fail-fast   Stop immediately on the first failed file

Behavior:
  - Recursively scans all .excalidraw.md files under --root
  - Replaces literal <br> tags with real line breaks
  - Writes files in place unless --dry-run is set`);
}

function collectExcalidrawMarkdownFiles(rootDir) {
  const files = [];

  function walk(currentDir) {
    for (const entry of fs.readdirSync(currentDir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) {
          walk(path.join(currentDir, entry.name));
        }
        continue;
      }

      if (entry.name.endsWith(".excalidraw.md")) {
        files.push(path.join(currentDir, entry.name));
      }
    }
  }

  walk(rootDir);
  return files;
}

function normalizeExcalidrawMarkdown(content) {
  return content.replace(/<br\s*\/?>(\s*)/gi, "\n$1");
}

function processFile(filePath, args) {
  const original = fs.readFileSync(filePath, "utf8");
  const updated = normalizeExcalidrawMarkdown(original);

  if (updated === original) {
    return false;
  }

  if (!args.dryRun) {
    fs.writeFileSync(filePath, updated, "utf8");
  }

  if (args.verbose) {
    console.log(`[changed] ${path.relative(process.cwd(), filePath)}`);
  }

  return true;
}

async function main() {
  const args = parseArgs(process.argv);
  const rootPath = path.resolve(args.root);

  if (!fs.existsSync(rootPath) || !fs.statSync(rootPath).isDirectory()) {
    fail(`Root folder does not exist: ${rootPath}`);
  }

  const files = collectExcalidrawMarkdownFiles(rootPath);
  const summary = {
    scannedFiles: files.length,
    touchedFiles: 0,
    failedFiles: 0,
    failures: [],
  };

  for (const filePath of files) {
    try {
      if (processFile(filePath, args)) {
        summary.touchedFiles += 1;
      }
    } catch (error) {
      summary.failedFiles += 1;
      summary.failures.push({
        filePath,
        error: error?.message || String(error),
      });

      if (args.verbose || args.failFast) {
        console.error(
          `[failed] ${path.relative(process.cwd(), filePath)}: ${error?.message || error}`,
        );
      }

      if (args.failFast) {
        throw error;
      }
    }
  }

  console.log(
    JSON.stringify(
      {
        root: rootPath,
        dryRun: args.dryRun,
        scannedFiles: summary.scannedFiles,
        touchedFiles: summary.touchedFiles,
        failedFiles: summary.failedFiles,
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