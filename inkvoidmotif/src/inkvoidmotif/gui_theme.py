"""Visual theme and browser-side accessibility helpers for inkvoidmotif Studio."""

from __future__ import annotations

import gradio as gr

CSS = r"""
:root {
  --ink: #252722;
  --pine: #435c4a;
  --paper: #f3efe3;
  --paper-deep: #e9e1cf;
  --cinnabar: #a84836;
}
html { color-scheme: only light !important; }
body, .gradio-container {
  background:
    radial-gradient(circle at 12% 2%, rgba(255,255,255,.92), transparent 28rem),
    linear-gradient(145deg, #f7f4ea 0%, #eee8da 100%) !important;
  color: var(--ink);
}
.gradio-container { box-sizing: border-box; max-width: 1500px !important; padding-top: 18px !important; }
.studio-hero {
  position: relative; overflow: hidden; border: 1px solid rgba(83,76,61,.2);
  border-radius: 24px; padding: 30px 34px; margin-bottom: 12px;
  background: linear-gradient(100deg, rgba(252,250,243,.96), rgba(229,225,209,.88));
  box-shadow: 0 16px 50px rgba(57,50,38,.08);
}
.eyebrow { color: var(--cinnabar); letter-spacing: .18em; font-size: 11px; font-weight: 750; text-transform: uppercase; }
.studio-hero h1 { margin: 4px 0 6px; color: #26352b; font-family: Georgia, "Noto Serif", serif; font-size: clamp(30px, 4vw, 48px); font-weight: 500; }
.studio-hero p { max-width: 790px; margin: 0; color: #696553; font-size: 15px; line-height: 1.65; }
.step-strip { display: flex; gap: 8px; margin: 10px 0 18px; flex-wrap: wrap; }
.step-chip { padding: 7px 12px; border-radius: 999px; border: 1px solid #d4ccb9; background: rgba(255,255,255,.55); color: #635e50; font-size: 12px; }
.step-chip b { color: var(--pine); margin-right: 5px; }
.stage { border: 1px solid rgba(84,77,63,.17) !important; border-radius: 18px !important; background: rgba(252,250,244,.78) !important; box-shadow: 0 8px 28px rgba(63,55,40,.045); }
.workflow-row { align-items: stretch !important; gap: 12px !important; }
.workflow-panel { min-width: 0 !important; }
.workflow-panel > .stage { box-sizing: border-box; height: 100%; }
.stage-title { margin: 0 0 4px !important; color: #314539 !important; font-family: Georgia, "Noto Serif", serif; }
.stage-title h2 { font-size: 20px !important; line-height: 1.2 !important; }
.compact-note { margin-top: -3px !important; }
.preview-stage { margin-top: 12px !important; padding-top: 4px !important; }
.preview-heading { margin: 2px 0 0 !important; color: #314539 !important; font-family: Georgia, "Noto Serif", serif; }
.quiet-note { color: #77705e; font-size: 12px; }
.status-card { border-left: 4px solid var(--cinnabar) !important; padding-left: 16px !important; background: rgba(255,251,242,.78); border-radius: 10px; }
.primary-action button, button.primary { background: var(--pine) !important; border-color: var(--pine) !important; color: white !important; }
.generate-action button { background: var(--cinnabar) !important; border-color: var(--cinnabar) !important; color: white !important; min-height: 48px; font-weight: 700; }
.preview-frame img { background: #e8e1d2 !important; object-fit: contain !important; }
.result-frame { border: 1px solid #c9bfaa !important; border-radius: 18px !important; overflow: hidden; }
.api-note { color: #746f61; font-size: 12px; text-align: right; }
footer { display: none !important; }
@media (max-width: 980px) {
  html, body, .gradio-container { box-sizing: border-box; width: 100% !important; min-width: 0 !important; max-width: 100% !important; margin: 0 !important; overflow-x: hidden !important; }
  .gradio-container { padding: 8px !important; }
  .gradio-container .row { flex-wrap: wrap !important; min-width: 0 !important; }
  .gradio-container .column { flex: 1 1 100% !important; width: 100% !important; min-width: 0 !important; max-width: 100% !important; }
  .gradio-container .form { min-width: 0 !important; }
  .workflow-row, .mobile-stack { flex-direction: column !important; flex-wrap: nowrap !important; width: 100% !important; min-width: 0 !important; }
  .workflow-panel, .preview-stage { box-sizing: border-box; flex: 1 1 100% !important; width: 100% !important; min-width: 0 !important; max-width: 100% !important; }
  .mobile-stack > * { box-sizing: border-box; flex: 1 1 auto !important; width: 100% !important; min-width: 0 !important; max-width: 100% !important; }
  .studio-hero { box-sizing: border-box; width: 100%; padding: 22px 16px; border-radius: 16px; }
  .studio-hero h1 { font-size: 29px; line-height: 1.12; }
  .stage { min-width: 0 !important; }
  .api-note { text-align: left; }
}
"""

HEAD = """<script>
  (() => {
    const url = new URL(window.location.href);
    if (!url.searchParams.has('__theme')) {
      url.searchParams.set('__theme', 'light');
      window.location.replace(url.toString());
      return;
    }
    const applyStatusA11y = () => {
      const status = document.getElementById('studio-status');
      if (status) {
        status.setAttribute('role', 'status');
        status.setAttribute('aria-live', 'polite');
        status.setAttribute('aria-atomic', 'true');
      }
    };
    window.addEventListener('load', applyStatusA11y);
    new MutationObserver(applyStatusA11y).observe(document.documentElement, {childList: true, subtree: true});
  })();
</script>"""


def theme() -> gr.Theme:
    """Return the Studio theme using only local/system font references."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        font=(
            gr.themes.LocalFont("IBM Plex Sans"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono=(gr.themes.LocalFont("IBM Plex Mono"), "ui-monospace", "monospace"),
    )
