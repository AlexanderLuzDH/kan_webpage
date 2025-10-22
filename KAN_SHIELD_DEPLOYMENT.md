# KAN Shield Landing Page — Deployment Guide

## Quick Start (Today)

### Option 1: Deploy HTML Only (Fastest)
The `kan-shield.html` file is self-contained (no backend needed initially).

1. **Upload to Netlify / Vercel:**
   ```bash
   git add website_copy/kan-shield.html
   git commit -m "Add KAN Shield landing page"
   git push
   ```
   - Form submissions auto-save to browser localStorage (JSON in console)
   - Once backend is ready, switch to real endpoint

2. **Custom Domain:**
   - Point `busleyden.com/kan-shield` → the HTML file (via redirect or CMS)

### Option 2: Add Zapier for 0-Code Email Capture (1 hour)

1. **Create Zapier Webhook:**
   - Go to https://zapier.com/apps/webhook/integrations/new
   - Create new Zap: Webhook → (trigger)
   - Copy webhook URL

2. **In HTML, replace localStorage with API call:**
   ```javascript
   // In kan-shield.html, update handleSubmit():
   const response = await fetch('https://hooks.zapier.com/hooks/catch/YOUR_ZAPIER_ID', {
       method: 'POST',
       body: JSON.stringify(data),
   });
   ```

3. **Zapier Actions (configure in Zap):**
   - Save to Google Sheet (email list)
   - Send email via Gmail
   - Post to Slack (internal alert)
   - Add to HubSpot / Pipedrive CRM

4. **Test:**
   - Fill out form on page → Zapier logs it → Google Sheet updated + email sent

### Option 3: Deploy Python Backend (Full Control, 2 hours)

1. **Install dependencies:**
   ```bash
   pip install flask sendgrid requests
   ```

2. **Set env vars (.env or shell):**
   ```bash
   export ZAPIER_WEBHOOK_URL="https://hooks.zapier.com/hooks/catch/YOUR_ID"
   export AIRTABLE_API_KEY="pat_..."
   export AIRTABLE_BASE_ID="appXXX"
   export SENDGRID_API_KEY="SG.xxx"
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
   ```

3. **Run locally:**
   ```bash
   python website_copy/kan-shield-backend.py
   ```

4. **Update HTML to POST to backend:**
   ```javascript
   // In handleSubmit(), replace fetch target:
   const response = await fetch('http://localhost:5000/api/submissions', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(data),
   });
   ```

5. **Deploy to production (Heroku, Railway, Render):**
   ```bash
   # Heroku example
   heroku login
   heroku create kan-shield-api
   git push heroku main
   heroku config:set ZAPIER_WEBHOOK_URL="..."
   ```

---

## Email Capture Flows

### Flow 1: Zapier Only (Recommended for MVP)
```
Form Submit → Zapier Webhook → [Google Sheet + Gmail + Slack]
```
- **Setup time:** 20 min
- **Cost:** Free (Zapier + Gmail)
- **Maintenance:** None

### Flow 2: Airtable + Zapier (CRM-Ready)
```
Form Submit → Python Backend → Airtable + Zapier → [CRM + Email + Slack]
```
- **Setup time:** 1 hour
- **Cost:** Airtable free ($12/mo if scaling)
- **Maintenance:** Monitor form submissions

### Flow 3: Custom Backend + Everything
```
Form Submit → Python Backend → Airtable + SendGrid + Slack + Zapier
```
- **Setup time:** 2 hours
- **Cost:** SendGrid ($20/mo), Slack, Zapier
- **Maintenance:** Backend uptime, API keys

---

## Environment Variables

| Var | Example | Where to Get |
|-----|---------|-------------|
| `ZAPIER_WEBHOOK_URL` | `https://hooks.zapier.com/...` | Zapier → New Zap |
| `AIRTABLE_API_KEY` | `pat_xxxxx` | Airtable Account → API |
| `AIRTABLE_BASE_ID` | `appXXX` | Airtable Base URL |
| `SENDGRID_API_KEY` | `SG.xxx` | SendGrid Dashboard |
| `SLACK_WEBHOOK_URL` | `https://hooks.slack.com/...` | Slack App → Webhooks |

---

## Testing

### Local Test (HTML only)
1. Open `kan-shield.html` in browser
2. Fill form → Submit
3. Open DevTools → Console → `localStorage.kan_shield_submissions` → JSON

### Local Test (with Backend)
```bash
curl -X POST http://localhost:5000/api/submissions \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jane Doe",
    "email": "jane@company.com",
    "company": "ACME Corp",
    "role": "CISO",
    "llm_provider": "OpenAI",
    "use_case": "Hallucination detection",
    "form_type": "trial"
  }'
```

---

## Deployment Checklist

- [ ] Upload `kan-shield.html` to Netlify / Vercel / GitHub Pages
- [ ] Set custom domain (`busleyden.com/kan-shield`)
- [ ] Create Zapier webhook (free tier)
- [ ] Configure Zapier actions (Google Sheet, Gmail, Slack)
- [ ] Test form submission end-to-end
- [ ] Add analytics tag (Google Analytics / Mixpanel)
- [ ] Monitor form submissions daily
- [ ] Reply to submissions within 24 hours
- [ ] After MVP: Deploy Python backend if scaling

---

## Quick Launch (Today)

1. **Push HTML to GitHub Pages (5 min):**
   ```bash
   cp website_copy/kan-shield.html public/index.html
   git push
   ```

2. **Create Zapier Webhook (5 min):**
   - Zapier → Create Zap → Webhook
   - Copy webhook URL

3. **Update HTML (2 min):**
   ```javascript
   // Change handleSubmit() to POST to Zapier
   ```

4. **Add Zapier Actions (10 min):**
   - Google Sheet (track emails)
   - Gmail (send confirmation)
   - Slack (internal alert)

5. **Test (5 min):**
   - Fill form → Check email → Check Slack

**Total: 27 minutes to launch.**

---

## Ongoing

- **Daily:** Check submissions (email/Slack)
- **Weekly:** Export CSV from Google Sheet; segment by company/role
- **Biweekly:** Reach out to interested prospects; schedule demos
- **Monthly:** Analyze conversion funnel; optimize messaging

---

## Future Enhancements

Once backend is live:
- A/B test CTAs (e.g., "Start Trial" vs "See Demo First")
- Track page scroll depth, time on page
- Add Drift chat for live sales engagement
- Segment audiences (by role, company size, LLM provider)
- Create targeted email sequences post-signup
- Integrate with HubSpot for lead scoring

