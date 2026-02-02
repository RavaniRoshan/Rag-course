export async function highlight(code: string, lang: string) {
  const { codeToHtml } = await import("shiki");
  return await codeToHtml(code, {
    lang,
    themes: {
      light: "github-light",
      dark: "github-dark-dimmed",
    },
  });
}
