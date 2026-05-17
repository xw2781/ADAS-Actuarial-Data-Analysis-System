const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const frontendRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(frontendRoot, "..");
const outputDir = path.join(frontendRoot, "build", "python_packages");
const pythonExe = process.env.PYTHON_EXE || process.env.PYTHON || "python";
const builder = path.join(repoRoot, "python-api", "tools", "build_wheel.py");

fs.rmSync(outputDir, { recursive: true, force: true });
fs.mkdirSync(outputDir, { recursive: true });

const result = spawnSync(pythonExe, [builder, "--out-dir", outputDir], {
  cwd: frontendRoot,
  stdio: "inherit",
  shell: false,
});

if (result.error) {
  console.error(result.error.message || result.error);
  process.exit(1);
}

process.exit(result.status || 0);
