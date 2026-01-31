/** @type {import('next').NextConfig} */
const nextConfig = {
  // Explicitly include shiki in the build output to ensure it is present
  // when using the dynamic import bypass (new Function).
  outputFileTracingIncludes: {
    "/**": ["./node_modules/shiki/**/*"],
  },
};

export default nextConfig;
