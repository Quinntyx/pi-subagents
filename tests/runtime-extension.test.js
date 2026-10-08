import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, realpathSync, copyFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import runtime, { configureInstalledRuntime } from "../extensions/runtime.js";

function installedPackage() {
  const root = mkdtempSync(join(tmpdir(), "pi-subagents-installed-"));
  mkdirSync(join(root, "src", "pi_subagents"), { recursive: true });
  writeFileSync(join(root, "pyproject.toml"), "[project]\nname='pi-subagents'\n");
  writeFileSync(join(root, "src", "pi_subagents", "__init__.py"), "");
  return root;
}

test("roots use this installed package, replacing stale environment overrides", () => {
  const root = installedPackage();
  try {
    const env = { PTC_SUBAGENTS_SOURCE: "/not-the-installed-package" };
    assert.equal(configureInstalledRuntime(env, root), realpathSync(root));
    assert.equal(env.PTC_SUBAGENTS_SOURCE, realpathSync(root));
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("children preserve their parent's source without probing or installing", () => {
  for (const policy of [{ PI_SUBAGENTS_DEPTH: "1" }, { PI_SUBAGENTS_PARENT_TOKEN: "parent" }]) {
    const env = { ...policy, PTC_SUBAGENTS_SOURCE: "/parent/installed-package" };
    assert.equal(configureInstalledRuntime(env, "/missing/package"), env.PTC_SUBAGENTS_SOURCE);
  }
});

test("malformed child policy remains available for strict validation", () => {
  const env = { PI_SUBAGENTS_DEPTH: "invalid", PTC_SUBAGENTS_SOURCE: "/parent/source" };
  configureInstalledRuntime(env, "/missing/package");
  assert.equal(env.PI_SUBAGENTS_DEPTH, "invalid");
  assert.equal(env.PTC_SUBAGENTS_SOURCE, "/parent/source");
});

test("incomplete installed packages fail explicitly", () => {
  const root = installedPackage();
  try {
    assert.throws(() => configureInstalledRuntime({}, join(root, "src")), /incomplete/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("Pi extension derives the runtime from its own module location", () => {
  const previous = process.env.PTC_SUBAGENTS_SOURCE;
  const depth = process.env.PI_SUBAGENTS_DEPTH;
  const parent = process.env.PI_SUBAGENTS_PARENT_TOKEN;
  try {
    delete process.env.PI_SUBAGENTS_DEPTH;
    delete process.env.PI_SUBAGENTS_PARENT_TOKEN;
    runtime();
    assert.equal(process.env.PTC_SUBAGENTS_SOURCE,
      realpathSync(fileURLToPath(new URL("../", import.meta.url))));
  } finally {
    for (const [key, value] of [["PTC_SUBAGENTS_SOURCE", previous], ["PI_SUBAGENTS_DEPTH", depth], ["PI_SUBAGENTS_PARENT_TOKEN", parent]]) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
});

test("relocated Pi cache loads without any development checkout", async () => {
  const root = installedPackage();
  try {
    mkdirSync(join(root, "extensions"));
    writeFileSync(join(root, "package.json"), '{"type":"module"}');
    copyFileSync(fileURLToPath(new URL("../extensions/runtime.js", import.meta.url)),
      join(root, "extensions", "runtime.js"));
    const relocated = await import(pathToFileURL(join(root, "extensions", "runtime.js")).href);
    const env = { HOME: join(root, "another-user-home") };
    assert.equal(relocated.configureInstalledRuntime(env), realpathSync(root));
    assert.equal(env.PTC_SUBAGENTS_SOURCE, realpathSync(root));
  } finally { rmSync(root, { recursive: true, force: true }); }
});
