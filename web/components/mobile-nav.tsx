"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion, AnimatePresence } from "framer-motion"
import { Menu, X, BookOpen, FileText } from "lucide-react"
import { cn } from "@/lib/utils"
import { Module } from "@/lib/content"

export function MobileNav({ modules }: { modules: Module[] }) {
  const [isOpen, setIsOpen] = React.useState(false)
  const pathname = usePathname()

  // Close menu when route changes
  React.useEffect(() => {
    setIsOpen(false)
  }, [pathname])

  // Prevent body scroll when menu is open
  React.useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden"
    } else {
      document.body.style.overflow = "unset"
    }
    return () => {
      document.body.style.overflow = "unset"
    }
  }, [isOpen])

  return (
    <div className="md:hidden mb-6">
      <button
        onClick={() => setIsOpen(true)}
        className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors border rounded-md bg-background/50 backdrop-blur supports-[backdrop-filter]:bg-background/60"
        aria-label="Open menu"
      >
        <Menu className="h-4 w-4" />
        <span>Menu</span>
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm"
            />

            {/* Drawer */}
            <motion.div
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", damping: 20, stiffness: 300 }}
              className="fixed inset-y-0 left-0 z-50 w-3/4 max-w-xs border-r bg-background p-6 shadow-xl overflow-y-auto"
              role="dialog"
              aria-modal="true"
              aria-label="Mobile navigation"
            >
              <div className="flex items-center justify-between mb-8">
                <span className="font-bold text-lg">Course Modules</span>
                <button
                  onClick={() => setIsOpen(false)}
                  className="rounded-md p-1 hover:bg-muted"
                  aria-label="Close menu"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <nav className="space-y-1">
                <Link
                  href="/introduction"
                  className={cn(
                    "flex items-center rounded-md px-2 py-2 text-sm font-medium hover:bg-muted hover:text-foreground transition-colors",
                    pathname === "/introduction"
                      ? "bg-muted text-foreground"
                      : "text-muted-foreground"
                  )}
                >
                  <BookOpen className="mr-2 h-4 w-4 shrink-0" />
                  Introduction
                </Link>
                {modules.map((module) => (
                  <Link
                    key={module.id}
                    href={`/${module.slug}`}
                    className={cn(
                      "flex items-center rounded-md px-2 py-2 text-sm font-medium hover:bg-muted hover:text-foreground transition-colors",
                      pathname === `/${module.slug}`
                        ? "bg-muted text-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <FileText className="mr-2 h-4 w-4 shrink-0" />
                    {module.title}
                  </Link>
                ))}
              </nav>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
