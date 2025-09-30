export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method Not Allowed' });
  }
  try {
    const { email, name = '', interest = '', source = 'unknown' } = req.body || {};
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(email))) {
      return res.status(400).json({ error: 'Invalid email' });
    }

    const to = process.env.EMAIL_TO || 'contact@busleyden.com';
    const from = process.env.FROM_EMAIL || 'onboarding@resend.dev';
    const apiKey = process.env.RESEND_API_KEY;

    const subject = name
      ? `KAN-Infinity Request — ${name}${interest ? ' (' + interest + ')' : ''}`
      : `KAN-Infinity Early Access`;

    const lines = [
      `Source: ${source}`,
      name ? `Name: ${name}` : null,
      `Email: ${email}`,
      interest ? `Interest: ${interest}` : null,
      `Time: ${new Date().toISOString()}`,
    ].filter(Boolean);
    const text = lines.join('\n');
    const html = `<pre style="font:14px/1.5 ui-sans-serif,system-ui">${lines
      .map((l) => l.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'))
      .join('\n')}</pre>`;

    if (!apiKey) {
      return res.status(500).json({ ok: false, error: 'Missing RESEND_API_KEY on server' });
    }

    const r = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ to, from, subject, text, html }),
    });
    if (!r.ok) {
      const body = await r.text();
      return res.status(502).json({ error: 'Email API failed', detail: body });
    }
    const data = await r.json();
    return res.status(200).json({ ok: true, id: data.id || null });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: String(err && err.message || err) });
  }
}
