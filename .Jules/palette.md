## 2025-02-19 - Accessible Code Copy Actions
**Learning:** On touch devices, hover-reveal actions for utilities like "Copy" are undiscoverable and often inaccessible.
**Action:** Always make utility actions visible or use a persistent tap-friendly trigger on mobile interfaces, rather than relying on `group-hover`.

## 2025-02-19 - Mobile Navigation Focus
**Learning:** Custom mobile drawers often lack proper semantic roles, confusing screen reader users who don't know they've entered a modal context.
**Action:** Always apply `role="dialog"`, `aria-modal="true"`, and `aria-label` to the drawer container to explicitly define the navigation context.
