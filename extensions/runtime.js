import { existsSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));

/** Pin roots to this installed package; children retain the parent's runtime. */
export function configureInstalledRuntime(env = process.env, source = packageRoot) {
  if (env.PI_SUBAGENTS_PARENT_TOKEN ||
      (env.PI_SUBAGENTS_DEPTH !== undefined && env.PI_SUBAGENTS_DEPTH !== "0")) {
    return env.PTC_SUBAGENTS_SOURCE;
  }
  const installed = realpathSync(source);
  if (!existsSync(join(installed, "pyproject.toml")) ||
      !existsSync(join(installed, "src", "pi_subagents", "__init__.py"))) {
    throw new Error(`Installed pi-subagents package is incomplete: ${installed}`);
  }
  env.PTC_SUBAGENTS_SOURCE = installed;
  return installed;
}

export default function () {
  configureInstalledRuntime();
}
