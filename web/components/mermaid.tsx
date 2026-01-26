"use client"

import React, { useEffect, useState } from 'react';
import mermaid from 'mermaid';
import { useTheme } from 'next-themes';

export function Mermaid({ chart }: { chart: string }) {
  const [svg, setSvg] = useState<string>('');
  const { theme } = useTheme();

  useEffect(() => {
    // Initialize mermaid with the correct theme
    mermaid.initialize({
      startOnLoad: false,
      theme: theme === 'dark' ? 'dark' : 'default',
      securityLevel: 'loose',
      fontFamily: 'var(--font-sans)',
    });

    const renderChart = async () => {
      try {
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        const { svg } = await mermaid.render(id, chart);
        setSvg(svg);
      } catch (error) {
        console.error('Failed to render mermaid chart:', error);
        // Optionally render the raw code if mermaid fails
      }
    };

    renderChart();
  }, [chart, theme]);

  return (
    <div
      className="mermaid my-6 flex justify-center bg-white dark:bg-zinc-900 p-4 rounded-lg border border-border overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
