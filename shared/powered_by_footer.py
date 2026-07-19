"""Shared Powered-by footer for agent /chat pages and static web."""

POWERED_BY_FOOTER_CSS = """
.site-powered-footer{margin-top:0;padding:16px 20px 20px;text-align:center;flex-shrink:0;
  border-top:1px solid var(--powered-footer-border, rgba(173,202,218,.25));
  background:var(--powered-footer-bg, rgba(255,255,255,.85))}
.site-powered-footer .powered-row{display:flex;flex-wrap:wrap;align-items:center;
  justify-content:center;gap:10px 18px}
.site-powered-footer .powered-label{font-size:11px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--powered-label-color, var(--text2, #64748b))}
.site-powered-footer .powered-logo-link{display:inline-flex;align-items:center;padding:6px 10px;
  border-radius:8px;line-height:0;
  background:var(--powered-logo-bg, #fff);
  border:1px solid var(--powered-logo-border, rgba(0,0,0,.08));
  box-shadow:var(--powered-logo-shadow, none);
  transition:transform .2s ease,box-shadow .2s ease}
.site-powered-footer .powered-logo-link:hover{
  transform:translateY(-1px);
  box-shadow:var(--powered-logo-shadow-hover, 0 2px 8px rgba(0,0,0,.06))}
.site-powered-footer .powered-logo{height:40px;width:auto;max-width:min(92vw,400px);object-fit:contain;display:block}
.site-powered-footer .lab-name{font-size:13px;font-weight:700;
  color:var(--powered-lab-color, var(--accent, #334155));letter-spacing:.02em}
"""

POWERED_BY_FOOTER_HTML = """
<footer class="site-powered-footer">
  <div class="powered-row">
    <span class="powered-label">Powered by</span>
    <a class="powered-logo-link"
       href="https://www.cityu.edu.hk/vetmedlife/"
       target="_blank"
       rel="noopener noreferrer"
       title="Jockey Club College of Veterinary Medicine and Life Sciences, CityU × Cornell">
      <img class="powered-logo"
           src="https://template.cityu.edu.hk/template/logo/JCC/CityU_Cornell_Horiz_Logo_CMYK.svg"
           alt="City University of Hong Kong Jockey Club College of Veterinary Medicine and Life Sciences in collaboration with Cornell University"
           loading="lazy"
           decoding="async">
    </a>
    <span class="lab-name">Smart Animal Management Lab</span>
  </div>
</footer>
"""
