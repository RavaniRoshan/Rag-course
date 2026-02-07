import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import Link from "next/link";
import { Analytics } from "@vercel/analytics/next";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAG Course",
  description: "Deep Dive into Retrieval Augmented Generation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen bg-background font-sans`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="relative flex min-h-screen flex-col">
            <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="container flex h-16 items-center justify-between px-4 md:px-6">
                <div className="flex items-center gap-3">
                  <Link className="flex items-center gap-2" href="/">
                    <span className="text-sm font-bold uppercase tracking-[0.3em]">Slush</span>
                    <span className="rounded-full border bg-muted/60 px-2 py-0.5 text-[10px] font-semibold uppercase text-muted-foreground">
                      Beta
                    </span>
                  </Link>
                </div>
                <div className="hidden items-center gap-6 text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground md:flex">
                  <Link className="transition-colors hover:text-foreground" href="#features">
                    Features
                  </Link>
                  <Link className="transition-colors hover:text-foreground" href="#onboarding">
                    Onboarding
                  </Link>
                  <Link className="transition-colors hover:text-foreground" href="#mobile">
                    Mobile
                  </Link>
                  <Link className="transition-colors hover:text-foreground" href="#testimonials">
                    Testimonials
                  </Link>
                  <Link className="transition-colors hover:text-foreground" href="#newsletter">
                    Newsletter
                  </Link>
                </div>
                <div className="flex items-center gap-3">
                  <Link
                    href="/introduction"
                    className="hidden h-10 items-center justify-center rounded-md border border-input bg-background px-4 text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground md:inline-flex"
                  >
                    Join waitlist
                  </Link>
                  <Link
                    href="/introduction"
                    className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-4 text-xs font-semibold uppercase tracking-[0.3em] text-primary-foreground transition-colors hover:bg-primary/90"
                  >
                    Get Slush
                  </Link>
                  <ThemeToggle />
                </div>
              </div>
            </header>
            <div className="flex-1">
              {children}
            </div>
          </div>
        </ThemeProvider>
        <Analytics />
      </body>
    </html>
  );
}
