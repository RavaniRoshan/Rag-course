"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Module } from "@/lib/content"
import { useState } from "react"
import { motion } from "framer-motion"
import { ChevronLeft, ChevronRight, BookOpen, FileText } from "lucide-react"

export function Sidebar({ modules }: { modules: Module[] }) {
  const pathname = usePathname()
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <motion.aside
      initial={{ width: 256 }}
      animate={{ width: isCollapsed ? 60 : 256 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="fixed top-14 z-30 hidden h-[calc(100vh-3.5rem)] shrink-0 border-r md:sticky md:block bg-background overflow-hidden"
    >
      <div className="flex items-center justify-between p-4 h-14 border-b border-border/40">
        {!isCollapsed && (
          <motion.h4
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-sm font-semibold whitespace-nowrap"
          >
            Course Modules
          </motion.h4>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={cn(
            "p-1.5 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors",
            isCollapsed && "mx-auto"
          )}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      <div className="p-2 space-y-1 overflow-y-auto h-[calc(100vh-8rem)]">
        <Link
            href="/introduction"
            className={cn(
              "group flex items-center rounded-md px-2 py-2 hover:bg-muted hover:text-foreground transition-colors",
              pathname === "/introduction"
                ? "bg-muted font-medium text-foreground"
                : "text-muted-foreground",
              isCollapsed ? "justify-center px-2" : ""
            )}
            title="Introduction"
        >
            <BookOpen className={cn("h-4 w-4 shrink-0", !isCollapsed && "mr-2")} />
            {!isCollapsed && <span className="text-sm truncate">Introduction</span>}
        </Link>

        {modules.map((item) => (
          <Link
            key={item.id}
            href={`/${item.slug}`}
            className={cn(
              "group flex items-center rounded-md px-2 py-2 hover:bg-muted hover:text-foreground transition-colors",
              pathname === `/${item.slug}`
                ? "bg-muted font-medium text-foreground"
                : "text-muted-foreground",
               isCollapsed ? "justify-center px-2" : ""
            )}
            title={item.title}
          >
             <FileText className={cn("h-4 w-4 shrink-0", !isCollapsed && "mr-2")} />
             {!isCollapsed && <span className="text-sm truncate">{item.title}</span>}
          </Link>
        ))}
      </div>
    </motion.aside>
  )
}
