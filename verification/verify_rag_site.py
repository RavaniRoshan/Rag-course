
from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()

        # 1. Landing Page
        print("Navigating to home...")
        page.goto("http://localhost:3002")
        expect(page).to_have_title("RAG Course")
        page.screenshot(path="verification/landing_page.png")
        print("Captured landing_page.png")

        # 2. Introduction
        print("Navigating to Introduction...")
        page.get_by_role("link", name="Introduction").click()
        expect(page.get_by_role("heading", name="Introduction")).to_be_visible()
        page.screenshot(path="verification/introduction_page.png")
        print("Captured introduction_page.png")

        # 3. Module 1 (with Mermaid)
        print("Navigating to Module 1...")
        page.get_by_role("link", name="Deep Dive: Module 1").click()
        expect(page.get_by_role("heading", name="Deep Dive: Module 1")).to_be_visible()
        # Wait for mermaid to render (it might take a moment if it's client-side)
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/module_1_mermaid.png")
        print("Captured module_1_mermaid.png")

        # 4. Theme Toggle
        print("Toggling theme...")
        # Assuming the toggle button is the moon/sun icon. It usually has an aria-label or we can find it by role.
        # In typical shadcn/ui it's a button with sr-only text "Toggle theme"
        try:
            theme_btn = page.get_by_role("button", name="Toggle theme")
            theme_btn.click()
            # Select 'Dark' or 'Light' from dropdown if it's a dropdown, or just click if it's a toggle.
            # Shadcn standard is a dropdown usually.
            if page.get_by_role("menuitem", name="Dark").is_visible():
                page.get_by_role("menuitem", name="Dark").click()

            page.wait_for_timeout(1000)
            page.screenshot(path="verification/theme_toggled.png")
            print("Captured theme_toggled.png")
        except Exception as e:
            print(f"Theme toggle interaction failed: {e}")

        browser.close()

if __name__ == "__main__":
    run_verification()
