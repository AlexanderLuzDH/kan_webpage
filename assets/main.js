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
const WELCOME = 'Welcome! — We will contact you shortly.';
const SUBMITTED = 'Submitted — Thanks! (email relay offline, recorded locally)';
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

function sendMail(to, subject, body){
  try {
    const href = `mailto:${to}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    const a = document.createElement('a');
    a.href = href; a.style.display = 'none';
    document.body.appendChild(a); a.click();
    setTimeout(()=>a.remove(), 0);
  } catch {}
}

// Hero early access form
const earlyForm = document.getElementById('early-access-form');
if (earlyForm) {
  earlyForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = /** @type {HTMLInputElement|null} */ (earlyForm.querySelector('input[name=email]'))?.value || '';
    if (!isValidEmail(email)) {
      showToast('err', 'Invalid Email', 'Please enter a valid email address.');
      return;
    }
    persist('early_access', { email });
    showToast('ok', 'Welcome!', 'We will contact you shortly.');
    sendMail('contact@busleyden.com', 'KAN-Infinity Early Access', `Email: ${email}\nSource: Hero form`);
    (earlyForm.querySelector('input[name=email]')).value = '';
  });
}

// Contact form
const contactForm = document.getElementById('contact-form');
if (contactForm) {
  contactForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const name = /** @type {HTMLInputElement|null} */ (contactForm.querySelector('#name'))?.value || '';
    const email = /** @type {HTMLInputElement|null} */ (contactForm.querySelector('#email'))?.value || '';
    const interest = /** @type {HTMLSelectElement|null} */ (contactForm.querySelector('#interest'))?.value || '';
    if (!name.trim() || !isValidEmail(email) || !interest.trim()) {
      showToast('warn', 'Missing Information', 'Please fill in all fields with valid information.');
      return;
    }
    persist('contact_requests', { name, email, interest });
    showToast('ok', 'Submitted', 'Thanks! We just opened your email draft.');
    const body = `Name: ${name}\nEmail: ${email}\nInterest: ${interest}\nSource: Contact form`;
    sendMail('contact@busleyden.com', `KAN-Infinity Request Access — ${interest}`, body);
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
