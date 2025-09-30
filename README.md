KAN‑Infinity Landing Page

Overview
- Static, responsive landing page using the content from `WEBSITE_TEXT.md`.
- Files: `index.html`, `assets/styles.css`, `assets/main.js`.

Run Locally
- Option 1: Open `index.html` directly in your browser.
- Option 2: Serve locally to enable smooth routing features:
  - With Node.js: `npx serve .` then open the printed URL.
  - With Python: `python -m http.server 8080` and open `http://localhost:8080`.

Structure
- Sections: overview, benefits, demos (placeholder), applications, how it works, frontiers, contact.
- Forms: Early access (email), Contact (name, email, interest). Both validate on the client and store submissions in `localStorage`.
- Toasts: Non‑intrusive notifications for validation and submission.

Notes
- The “Docs” nav link is a placeholder (`#`) since no docs URL or content was provided in `WEBSITE_TEXT.md`.
- All text strings follow `WEBSITE_TEXT.md`. Minor typographic fixes were applied (en‑dashes, × symbol).

Images (Demos/Results)
- Place PNGs in `assets/img/` using these filenames:
  - `cifar10_zero_shot_bar.png`
  - `cifar10_fewshot_fusion_bars.png`
  - `cifar10_accuracy_vs_coverage.png`
  - `cifar10_next16_tile_L5.png`
  - `cifar10_next16_tile_L10.png`
  - `agnews_label_efficiency.png`
  - `agnews_5_labels_bar.png`
- Open the page and click any thumbnail to view in a lightbox. If an image is missing, its tile will simply show a blank background.

Serverless email (Vercel)
- The forms POST to `/api/contact` which sends an email via SendGrid (preferred) or Resend (fallback).

Option A — SendGrid (no DNS changes)
- Create free SendGrid account → Settings → Sender Authentication → Single Sender → verify `FROM_EMAIL`.
- Create API key (Mail Send Full Access).
- Set env vars in Vercel → Project → Settings → Environment Variables:
  - `SENDGRID_API_KEY` — your key
  - `FROM_EMAIL` — the verified single sender (e.g., `contact@busleyden.com`)
  - `EMAIL_TO` — where to receive submissions
- Redeploy and test.

Option B — Resend (requires domain or limited tests)
- Set env vars:
  - `RESEND_API_KEY`
  - `EMAIL_TO`
  - `FROM_EMAIL` — must be from a verified domain in Resend
- Redeploy and test.
