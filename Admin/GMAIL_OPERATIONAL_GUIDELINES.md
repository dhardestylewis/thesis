# Gmail Operational Guidelines (GUIDELINES.md)

This document defines the strict operational boundaries and best practices for the AI Agent when interacting with Gmail. These rules are designed to protect the account from bans, prevent spam classification, and ensure user control.

## 🛡️ Core Safety Principles

1. **Human-in-the-loop (HITL):** NO email shall be sent without explicit manual confirmation ("Send", "Yes", "Confirm") from the user.
2. **Draft First:** All outbound communications must be created as drafts first for review, unless the user explicitly requests an immediate send for a specific, pre-verified message.
3. **No Impersonation:** Never attempt to impersonate other domains or senders. The `From` header must always represent the authenticated user.
4. **Volume Control:** Avoid "burst" sending. If multiple emails need to be sent, space them out (e.g., 1 message every 30-60 seconds) to mimic human behavior and avoid rate limits.
5. **Credential Safety:** NEVER delete `credentials.json` or `token.json` unless explicitly commanded to "delete files". Interpreting "close access" or "disconnect" means stopping processes/ports, NOT deleting authentication files.

## 📧 Drafting Guidelines

- **Clarity & Purpose:** Every email must have a clear, descriptive subject line and a concise body.
- **Consistent Category:** Keep the content of a single email focused on one category (e.g., don't mix a promotional message with a receipt).
- **No Automated "Blasts":** Do not use this tool for bulk marketing or unsolicited outreach. It is for personalized, intentional communication.

## 🎭 Mimicry & Randomization Specifics

To avoid "bot" fingerprinting, all automated actions must incorporate stochastic (randomized) elements:

1. **The "Jitter" Rule:** Never use fixed intervals. Every delay or scheduled action must include a **30-50% jitter**.
2. **Burst vs. Drip Pattern:** Human behavior is "bursty." Group tasks (e.g., summarizing 5 emails) then remain idle (30m+) rather than constant polling.
3. **Contextual Thinking Time:** drafting should follow a delay scaled to message length (~1s per 5 words).
4. **Timezone Alignment:** Primary operations shall occur during active hours (**8 AM - 10 PM EST**).

## 🧠 Advanced Human-Like Logic

1. **Non-Uniform Jitter:** Use a **Normal (Gaussian) Distribution** for delays rather than uniform ranges.
2. **Interaction Chaining:** Simulate navigation: *List -> Pause -> Read -> Pause -> Action.*
3. **Task Fatigue:** Increase idle time proportionally to the number of tasks performed in a burst.
4. **No 0-ms Latency:** Enforce a minimum 300ms-800ms "UI travel" delay between all API calls.

## 🛠️ Technical Implementation Details (For Future Agents)

1. **Draft Deletion Procedure:** The MCP server lacks a `delete_draft` tool. To delete a draft:
   - Search for the draft (e.g., by subject) to find its underlying **Message ID**.
   - Use `modify_gmail_message_labels` to add the `TRASH` label to that Message ID.
2. **Local Activity Logging:** Maintain a `C:\Users\dhl\.gemini\antigravity\scratch\gmail-mcp-workspace\ACTIVITY_LOG.json` to track:
   - Draft IDs created by this agent.
   - Message IDs of sent or trashed items.
   - Timestamps of all operations.
3. **Agent-Specific Scope:** Only manage or delete drafts identified in the `ACTIVITY_LOG.json`. Never touch personal/manual drafts.

## 🛡️ Defense-in-Depth

1. **Exponential Backoff:** On API failure, use base-2 backoff with jitter (2s, 4s, 8s...).
2. **Metadata Hygiene:** Avoid identical templates across multiple recipients; vary greetings and phrasing slightly.
3. **"Human Breadcrumb" Pattern:** Occasionally perform read-only "Neutral Actions" (checking labels/counts) with no subsequent write action.

##  Quotas & Limits

- **Daily Sending Limit:** Aim for **<50 per day** (Safe margin vs 500/2000 Google limits).
- **Rate Limiting:** Max 1-2 calls/sec for metadata, 1 per 30s for sending.
- **Spam Threshold:** Target <0.1% reported spam.

## 📝 User Verification Checklist

Before "Send" approval, present:
1. Recipient(s) | 2. Subject | 3. Draft ID | 4. Summary of Intent

---
*Created: 2026-01-22*

## 🚀 Technical Setup & Initialization

If the Gmail integration is not active or credentials are missing:

1.  **Acquire Credentials:**
    -   Create/Select a Google Cloud Project with **Gmail API** enabled.
    -   Configure **OAuth Consent Screen** (User Type: External, Add yourself as Test User).
    -   Create **OAuth Client ID** (Desktop app) credentials.
    -   Download JSON, rename to `credentials.json`, and place in:
        `C:\Users\dhl\.gemini\antigravity\scratch\gmail-mcp-workspace\`

2.  **Dependencies:**
    -   Ensure `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib` are installed.

3.  **Run Initialization:**
4.  **Draft Creation via Python API (Preferred Method):**
    -   Use `create_draft_with_auth.py` in the workspace.
    -   **Scopes Required:**
        -   `https://www.googleapis.com/auth/gmail.readonly`
        -   `https://www.googleapis.com/auth/gmail.compose`
    -   **Credentials Location:** `C:\Users\dhl\.gemini\antigravity\scratch\gmail-mcp-workspace\credentials.json`
    -   **Token Location:** `token.json` (auto-generated in same dir).
    -   *Note:* If you get "403 Insufficient Permission", delete `token.json` and re-run the script to re-authenticate with new scopes.

5.  **Helper Utilities:**
    -   `create_interview_emails_pdf.py`: Compiles local `.eml` files into a PDF for review.
    -   `archive_verbatim.py`: Downloads recent emails for backup/read-only access.
    
    *Verified Workflow (Jan 23, 2026):*
    1. Check `credentials.json` exists in workspace.
    2. Run `python create_draft_with_auth.py`.
    3. Authenticate in browser.
    4. Draft appears in Gmail Drafts folder.

    *Attachments Available:*
    -   `Submitted\IRB_Submitted\Protocol_and_Submission\Predicting_NIMBYism_Interview_Guide.pdf` (Primary)
    -   `Thesis_Draft\Project_One_Pager\Project_One_Pager.pdf` (Optional Project Summary)
