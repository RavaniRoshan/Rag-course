import fs from 'fs';
import path from 'path';

// Adjust path to point to the modules directory relative to the web directory
const MODULES_DIR = path.join(process.cwd(), '..', 'modules');

export interface Module {
  id: string;
  slug: string;
  title: string;
  dirName: string;
}

export interface ModuleContent extends Module {
  markdown: string;
  codeFiles: { name: string; content: string; language: string }[];
}

export async function getModules(): Promise<Module[]> {
  if (!fs.existsSync(MODULES_DIR)) {
    console.warn(`Modules directory not found at ${MODULES_DIR}`);
    return [];
  }

  const entries = fs.readdirSync(MODULES_DIR, { withFileTypes: true });

  const modules = entries
    .filter(entry => entry.isDirectory() && entry.name.startsWith('module_'))
    .map(entry => {
      const dirPath = path.join(MODULES_DIR, entry.name);
      const readmePath = path.join(dirPath, 'README.md');
      let title = entry.name
        .replace(/^module_\d+_/, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase()); // Fallback title

      if (fs.existsSync(readmePath)) {
        const content = fs.readFileSync(readmePath, 'utf-8');
        const match = content.match(/^#\s+(.+)$/m);
        if (match) {
          const raw = match[1].trim();
          title = raw.replace(/[*_]+/g, '');
        }
      }

      return {
        id: entry.name,
        slug: entry.name,
        title,
        dirName: entry.name,
      };
    })
    .sort((a, b) => a.id.localeCompare(b.id)); // Sort by folder name (module_01, module_02, etc.)

  return modules;
}

export async function getModule(slug: string): Promise<ModuleContent | null> {
  if (!fs.existsSync(MODULES_DIR)) {
    return null;
  }

  const modules = await getModules();
  const foundModule = modules.find(m => m.slug === slug);

  if (!foundModule) {
    return null;
  }

  const dirPath = path.join(MODULES_DIR, foundModule.dirName);
  const readmePath = path.join(dirPath, 'README.md');

  let markdown = '';
  if (fs.existsSync(readmePath)) {
    markdown = fs.readFileSync(readmePath, 'utf-8');
  }

  // Get all code files (.py)
  const codeFiles: { name: string; content: string; language: string }[] = [];
  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    if (file.endsWith('.py')) {
      const content = fs.readFileSync(path.join(dirPath, file), 'utf-8');
      codeFiles.push({
        name: file,
        content,
        language: 'python'
      });
    }
  }

  return {
    ...foundModule,
    markdown,
    codeFiles
  };
}

export async function getCourseReadme(): Promise<string> {
    const readmePath = path.join(MODULES_DIR, '..', 'README.md');
    if (fs.existsSync(readmePath)) {
        return fs.readFileSync(readmePath, 'utf-8');
    }
    return '';
}
