"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Module } from "@/lib/content"

export function Sidebar({ modules }: { modules: Module[] }) {
  const pathname = usePathname()

  return (
    <aside className="fixed top-14 z-30 -ml-2 hidden h-[calc(100vh-3.5rem)] w-64 shrink-0 overflow-y-auto border-r md:sticky md:block py-6 pr-4 lg:py-8">
      <h4 className="mb-4 rounded-md px-2 py-1 text-sm font-semibold">
        Course Modules
      </h4>
      <div className="grid grid-flow-row auto-rows-max text-sm space-y-1">
        {modules.map((item) => (
          <Link
            key={item.id}
            href={`/${item.slug}`}
            className={cn(
              "group flex w-full items-center rounded-md border border-transparent px-2 py-1.5 hover:bg-muted hover:text-foreground",
              pathname === `/${item.slug}`
                ? "font-medium text-foreground bg-muted"
                : "text-muted-foreground"
            )}
          >
            {item.title}
          </Link>
        ))}
      </div>
    </aside>
  )
}
