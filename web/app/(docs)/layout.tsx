import { Sidebar } from "@/components/sidebar";
import { MobileNav } from "@/components/mobile-nav";
import { getModules } from "@/lib/content";

export default async function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const modules = await getModules();

  return (
    <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10 px-4 md:px-6">
      <Sidebar modules={modules} />
      <main className="relative py-6 lg:gap-10 lg:py-8">
        <MobileNav modules={modules} />
        <div className="mx-auto w-full min-w-0 max-w-4xl">
          {children}
        </div>
      </main>
    </div>
  );
}
