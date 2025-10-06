/* Navigation */
const navToggle = document.querySelector('.nav-toggle');
const navList = document.getElementById('site-nav');
if (navToggle && navList) {
  navToggle.addEventListener('click', () => {
    const open = navList.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', String(open));
  });
}

/* Active section highlighting */
const links = Array.from(document.querySelectorAll('.nav-list a[href^="#"]'));
const sections = links
  .map(a => a.getAttribute('href'))
  .filter(Boolean)
  .map(id => document.querySelector(id))
  .filter(Boolean);

const io = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    const id = '#' + entry.target.id;
    const link = links.find(l => l.getAttribute('href') === id);
    if (!link) return;
    if (entry.isIntersecting) {
      links.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    }
  });
}, { rootMargin: '-40% 0px -55% 0px', threshold: 0.01 });

sections.forEach(sec => io.observe(sec));

/* Toasts */
const toasts = document.getElementById('toasts');
function showToast(kind, title, message, timeout = 4200) {
  if (!toasts) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = `<div><strong>${title}</strong><div>${message}</div></div><button class="close" aria-label="Dismiss">×</button>`;
  el.querySelector('.close')?.addEventListener('click', () => el.remove());
  toasts.appendChild(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
}

/* Forms: validation + local record */
const EMAIL_INVALID = 'Invalid Email — Please enter a valid email address.';
const WELCOME = 'Thank You! — We will contact you shortly.';
const SUBMITTED = 'Thank You! — We will contact you shortly.';
const MISSING = 'Missing Information — Please fill in all fields with valid information.';
const THANKS = 'Thank You! — We will contact you within 24 hours.';

function isValidEmail(v) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v.trim());
}

function persist(key, data){
  try {
    const prev = JSON.parse(localStorage.getItem(key) || '[]');
    prev.push({...data, ts: new Date().toISOString()});
    localStorage.setItem(key, JSON.stringify(prev));
  } catch {}
}

async function postJSON(url, payload){
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data?.ok === false) {
    throw new Error(data?.error || 'Request failed');
  }
  return data;
}

// Hero early access form
const earlyForm = document.getElementById('early-access-form');
if (earlyForm) {
  earlyForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = /** @type {HTMLInputElement|null} */ (earlyForm.querySelector('input[name=email]'))?.value || '';
    if (!isValidEmail(email)) {
      showToast('err', 'Invalid Email', 'Please enter a valid email address.');
      return;
    }
    persist('early_access', { email });
    try {
      await postJSON('/api/contact', { source: 'hero', email });
      showToast('ok', 'Thank You!', 'We will contact you shortly.');
    } catch {
      showToast('warn', 'Saved locally', 'Network issue; we will follow up.');
    }
    (earlyForm.querySelector('input[name=email]')).value = '';
  });
}

// Contact form
const contactForm = document.getElementById('contact-form');
if (contactForm) {
  contactForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = /** @type {HTMLInputElement|null} */ (contactForm.querySelector('#name'))?.value || '';
    const email = /** @type {HTMLInputElement|null} */ (contactForm.querySelector('#email'))?.value || '';
    const interest = /** @type {HTMLSelectElement|null} */ (contactForm.querySelector('#interest'))?.value || '';
    if (!name.trim() || !isValidEmail(email) || !interest.trim()) {
      showToast('warn', 'Missing Information', 'Please fill in all fields with valid information.');
      return;
    }
    persist('contact_requests', { name, email, interest });
    try {
      await postJSON('/api/contact', { source: 'contact', name, email, interest });
      showToast('ok', 'Thank You!', 'We will contact you within 24 hours.');
    } catch {
      showToast('warn', 'Saved locally', 'Network issue; we will follow up.');
    }
    contactForm.reset();
  });
}

/* Smooth close nav on link click (mobile) */
links.forEach(a => a.addEventListener('click', () => {
  if (navList?.classList.contains('open')) {
    navList.classList.remove('open');
    navToggle?.setAttribute('aria-expanded', 'false');
  }
}));

/* Lightbox for gallery */
const gallery = document.querySelector('.gallery');
if (gallery) {
  const lb = document.createElement('div');
  lb.className = 'lightbox';
  lb.innerHTML = '<button class="close" aria-label="Close">Esc</button><figure><img alt="" /><figcaption class="caption"></figcaption></figure>';
  document.body.appendChild(lb);
  const imgEl = lb.querySelector('img');
  const capEl = lb.querySelector('.caption');
  const close = () => lb.classList.remove('open');
  lb.querySelector('.close')?.addEventListener('click', close);
  lb.addEventListener('click', (e) => { if (e.target === lb) close(); });
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  gallery.addEventListener('click', (e) => {
    const a = /** @type {HTMLElement} */(e.target)?.closest('a');
    if (!a) return;
    e.preventDefault();
    const href = a.getAttribute('href');
    const caption = a.getAttribute('data-caption') || '';
    if (href && imgEl) {
      imgEl.src = href;
      if (capEl) capEl.textContent = caption;
      lb.classList.add('open');
    }
  });
}

/* ROI Calculator */
function fmtUSD(n){
  try { return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(n); } catch { return `$${n}` }
}
const roiForm = document.getElementById('roi-form');
if (roiForm && typeof roiEstimate === 'function'){
  const out = document.getElementById('roi-out');
  const run = () => {
    const volume = Number(document.getElementById('roi-volume').value||0);
    const legacyReviewRate = Number(document.getElementById('roi-legacy').value||0)/100;
    const ourCoverage = Number(document.getElementById('roi-coverage').value||0)/100;
    const costPerReview = Number(document.getElementById('roi-review').value||0);
    const pricePerCertified = Number(document.getElementById('roi-price').value||0);
    const alpha = Number(document.getElementById('roi-alpha')?.value||5)/100;
    const conf = Number(document.getElementById('roi-conf')?.value||95)/100;
    const r = roiEstimate({ volume, legacyReviewRate, ourCoverage, costPerReview, pricePerCertified});
    if (out){
      out.innerHTML = `
        <article class="card"><h3>Certified/month</h3><p>${r.certified.toLocaleString()}</p></article>
        <article class="card"><h3>Abstentions/month</h3><p>${r.abstain.toLocaleString()}</p></article>
        <article class="card"><h3>Spend</h3><p>${fmtUSD(r.monthlySpend)}</p></article>
        <article class="card"><h3>Review Cost</h3><p>${fmtUSD(r.monthlyReviewCost)}</p></article>
        <article class="card"><h3>Estimated Savings</h3><p><strong>${fmtUSD(r.monthlySavings)}</strong></p></article>
        <article class="card"><h3>Target</h3><p>Accepted-subset error ≤ ${(alpha*100).toFixed(1)}% with confidence ≥ ${(conf*100).toFixed(1)}%.</p></article>`;
    }
  };
  document.getElementById('roi-run')?.addEventListener('click', run);
  run();
}
