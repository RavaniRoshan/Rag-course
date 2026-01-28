/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: ["shiki", "@shikijs/core", "vscode-oniguruma", "vscode-textmate"],
  experimental: {
    serverComponentsExternalPackages: ["shiki", "@shikijs/core", "vscode-oniguruma", "vscode-textmate"],
  },
};

export default nextConfig;
