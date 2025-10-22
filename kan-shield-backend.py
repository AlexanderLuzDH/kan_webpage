"""
KAN Shield Landing Page Backend
Receives form submissions and routes them to:
- Airtable (email capture DB)
- Email (via SendGrid or Mailgun)
- Zapier Webhook (for auto-routing to CRM, Slack, etc.)
"""

from flask import Flask, request, jsonify
import os
import json
from datetime import datetime
import requests

app = Flask(__name__)

# Config (set via env vars or .env)
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY", "")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "")
AIRTABLE_TABLE_NAME = "KAN Shield Signups"

ZAPIER_WEBHOOK_URL = os.environ.get("ZAPIER_WEBHOOK_URL", "")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


@app.route("/api/submissions", methods=["POST"])
def handle_submission():
    """
    Accept form submissions from kan-shield.html
    Body: {
        name: string,
        email: string,
        company: string,
        role?: string,
        llm_provider?: string,
        use_case?: string,
        preferred_time?: string,  // for demo requests
        form_type: "trial" | "demo"  // added by JS
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not all(k in data for k in ["name", "email", "company"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Add metadata
        data["timestamp"] = datetime.utcnow().isoformat()
        data["form_type"] = data.get("form_type", "trial")
        
        # 1. Save to Airtable (email capture)
        if AIRTABLE_API_KEY and AIRTABLE_BASE_ID:
            save_to_airtable(data)
        
        # 2. Send to Zapier webhook (auto-routing, CRM, Slack, email)
        if ZAPIER_WEBHOOK_URL:
            send_to_zapier(data)
        
        # 3. Send confirmation email (optional)
        if SENDGRID_API_KEY:
            send_confirmation_email(data)
        
        # 4. Post to Slack (internal alert)
        if SLACK_WEBHOOK_URL:
            post_to_slack(data)
        
        return jsonify({
            "status": "success",
            "message": "Submission received. We'll be in touch within 24 hours.",
            "submission_id": data["timestamp"]
        }), 200
    
    except Exception as e:
        print(f"Error handling submission: {e}")
        return jsonify({"error": str(e)}), 500


def save_to_airtable(data):
    """Save to Airtable for email capture and CRM."""
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }
    
    record = {
        "fields": {
            "Name": data.get("name", ""),
            "Email": data.get("email", ""),
            "Company": data.get("company", ""),
            "Role": data.get("role", ""),
            "LLM Provider": data.get("llm_provider", ""),
            "Use Case": data.get("use_case", ""),
            "Preferred Time": data.get("preferred_time", ""),
            "Form Type": data.get("form_type", "trial"),
            "Timestamp": data.get("timestamp", ""),
        }
    }
    
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    requests.post(url, json=record, headers=headers)


def send_to_zapier(data):
    """Send to Zapier for workflow automation (CRM, email, Slack, etc.)"""
    payload = {
        "name": data.get("name"),
        "email": data.get("email"),
        "company": data.get("company"),
        "role": data.get("role"),
        "llm_provider": data.get("llm_provider"),
        "use_case": data.get("use_case"),
        "preferred_time": data.get("preferred_time"),
        "form_type": data.get("form_type"),
        "timestamp": data.get("timestamp"),
    }
    requests.post(ZAPIER_WEBHOOK_URL, json=payload)


def send_confirmation_email(data):
    """Send confirmation email via SendGrid."""
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    
    try:
        message = Mail(
            from_email="noreply@busleyden.com",
            to_emails=data["email"],
            subject="KAN Shield Trial — Setup Instructions Coming",
            html_content=f"""
            <p>Hi {data.get('name', 'there')},</p>
            <p>Thanks for your interest in KAN Shield! We've received your request for a free 14-day trial.</p>
            <p><strong>What's next:</strong></p>
            <ul>
                <li>We'll send setup instructions within 24 hours.</li>
                <li>Week 1: Shadow mode (no user impact, just logs).</li>
                <li>Week 2: Enforce on one low-risk route (measure impact).</li>
                <li>Track incidents prevented, FPR/FNR, and p95 latency via your dashboard.</li>
            </ul>
            <p>Questions? Reply to this email or call our team.</p>
            <p>Best,<br/>Busleyden Team</p>
            """
        )
        
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
    except Exception as e:
        print(f"Error sending confirmation email: {e}")


def post_to_slack(data):
    """Post to Slack for internal awareness (sales/ops)."""
    message = {
        "text": f"New KAN Shield {data.get('form_type', 'trial')} request!",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{data.get('name')}* from *{data.get('company')}*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Email:*\n{data.get('email')}"},
                    {"type": "mrkdwn", "text": f"*Role:*\n{data.get('role', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*LLM Provider:*\n{data.get('llm_provider', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Type:*\n{data.get('form_type', 'trial')}"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Use Case:*\n{data.get('use_case', 'Not provided')}"
                }
            }
        ]
    }
    
    requests.post(SLACK_WEBHOOK_URL, json=message)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
