/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: [
    "shiki",
    "@shikijs/core",
    "@shikijs/vscode-textmate",
    "@shikijs/engine-oniguruma"
  ],
};

export default nextConfig;
