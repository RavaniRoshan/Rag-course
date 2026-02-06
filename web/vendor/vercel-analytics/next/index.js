import Script from "next/script";

export function Analytics() {
  return (
    <Script
      src="https://va.vercel-scripts.com/v1/script.js"
      strategy="afterInteractive"
    />
  );
}
