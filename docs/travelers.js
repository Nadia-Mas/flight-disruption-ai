(() => {
  function mountTravelers() {
    const hero = document.querySelector('.hero');
    if (!hero || document.querySelector('.traveler-strip')) return;

    const style = document.createElement('style');
    style.textContent = `
      .traveler-strip{margin:-6px 0 26px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
      .traveler-card{position:relative;overflow:hidden;min-height:104px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,color-mix(in srgb,var(--brand2) 7%,var(--surface)),var(--surface));box-shadow:var(--shadow);padding:18px 20px;display:flex;align-items:center;gap:18px}
      .traveler-figure{font-size:40px;line-height:1;filter:drop-shadow(0 6px 10px rgba(17,58,91,.12));animation:travelerBob 2.6s ease-in-out infinite;transform-origin:center bottom}
      .traveler-card.business .traveler-figure{animation-delay:.8s}
      .traveler-copy{min-width:0}
      .traveler-copy strong{display:block;font-size:14px;color:var(--text);margin-bottom:5px}
      .traveler-copy span{display:block;font-size:11px;line-height:1.5;color:var(--muted)}
      .traveler-path{position:absolute;left:0;right:0;bottom:0;height:3px;background:linear-gradient(90deg,transparent,color-mix(in srgb,var(--brand2) 45%,transparent),transparent);opacity:.55}
      .traveler-plane{position:absolute;right:16px;top:12px;font-size:18px;opacity:.28;animation:travelerPlane 6s ease-in-out infinite}
      .traveler-card.business .traveler-plane{animation-delay:1.2s}
      @keyframes travelerBob{0%,100%{transform:translateY(0) rotate(-1deg)}50%{transform:translateY(-5px) rotate(1deg)}}
      @keyframes travelerPlane{0%,100%{transform:translate(0,0) rotate(-8deg)}50%{transform:translate(-18px,-5px) rotate(3deg)}}
      @media(max-width:700px){.traveler-strip{grid-template-columns:1fr}.traveler-card{min-height:92px}.traveler-figure{font-size:34px}}
      @media(prefers-reduced-motion:reduce){.traveler-figure,.traveler-plane{animation:none!important}}
    `;
    document.head.appendChild(style);

    const strip = document.createElement('section');
    strip.className = 'traveler-strip';
    strip.setAttribute('aria-label', 'FlightRescue passenger personas');
    strip.innerHTML = `
      <article class="traveler-card family">
        <div class="traveler-figure" aria-hidden="true">👨‍👩‍👧‍👦🧳</div>
        <div class="traveler-copy"><strong>Family trip</strong><span>Parents and kids trying to get home safely when delays and cancellations keep changing the plan.</span></div>
        <span class="traveler-plane" aria-hidden="true">✈️</span><i class="traveler-path"></i>
      </article>
      <article class="traveler-card business">
        <div class="traveler-figure" aria-hidden="true">🧑‍💼💼</div>
        <div class="traveler-copy"><strong>Business traveler</strong><span>A passenger who needs to know whether a disruption could affect an important meeting or connection.</span></div>
        <span class="traveler-plane" aria-hidden="true">✈️</span><i class="traveler-path"></i>
      </article>`;
    hero.insertAdjacentElement('afterend', strip);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountTravelers);
  else mountTravelers();
})();
