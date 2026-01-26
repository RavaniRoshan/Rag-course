import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["shiki", "@shikijs/core", "vscode-oniguruma", "vscode-textmate"],
};

export default nextConfig;
