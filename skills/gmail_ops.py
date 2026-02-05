import base64
import json
import os
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Dict, List, Optional

from error_util import ToolError, catch_errors

__tool_prefix__ = "gmail"


def _require_google_api() -> Dict[str, Any]:
    try:
        import google  # noqa: F401
        return {"success": True}
    except Exception:
        return {
            "success": False,
            "error": "google api libs not installed",
            "install_commands": [
                "pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2"
            ],
        }


def _get_paths() -> Dict[str, str]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    secrets_dir = os.path.join(repo_root, "secrets")
    data_dir = os.path.join(repo_root, "data")
    return {
        "client_secret": os.getenv("GMAIL_CLIENT_SECRET_PATH", os.path.join(secrets_dir, "gmail_client_secret.json")),
        "token": os.getenv("GMAIL_TOKEN_PATH", os.path.join(secrets_dir, "gmail_token.json")),
        "triage_rules": os.getenv("GMAIL_TRIAGE_RULES_PATH", os.path.join(data_dir, "gmail_triage_rules.json")),
        "outbox": os.getenv("GMAIL_OUTBOX_PATH", os.path.join(data_dir, "gmail_outbox.json")),
    }


def _get_scopes(scopes: Optional[List[str]] = None) -> List[str]:
    if scopes:
        return scopes
    env_scopes = os.getenv("GMAIL_SCOPES", "").strip()
    if env_scopes:
        return [s.strip() for s in env_scopes.split(",") if s.strip()]
    return [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
    ]


def _load_credentials(scopes: Optional[List[str]] = None):
    check = _require_google_api()
    if not check["success"]:
        raise ToolError(check["error"], code="DEPS_MISSING", context=check)

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    paths = _get_paths()
    token_path = paths["token"]
    if not os.path.exists(token_path):
        raise ToolError(
            f"Gmail token not found at {token_path}",
            code="NO_TOKEN",
            context={"token_path": token_path},
        )

    creds = Credentials.from_authorized_user_file(token_path, _get_scopes(scopes))
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    if not creds or not creds.valid:
        raise ToolError("Gmail credentials invalid", code="INVALID_TOKEN")
    return creds


def _build_service(creds):
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=creds)


def _load_json_file(path: str, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json_file(path: str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _label_map(service) -> Dict[str, str]:
    resp = service.users().labels().list(userId="me").execute()
    return {l["name"]: l["id"] for l in (resp.get("labels") or [])}


def _ensure_labels(service, label_names: List[str]) -> Dict[str, str]:
    name_to_id = _label_map(service)
    created = {}
    for name in label_names:
        if name in name_to_id:
            continue
        body = {"name": name}
        try:
            label = service.users().labels().create(userId="me", body=body).execute()
            name_to_id[name] = label.get("id")
            created[name] = label.get("id")
        except Exception:
            continue
    return name_to_id


def _get_header(payload: Dict[str, Any], name: str) -> str:
    for header in payload.get("headers") or []:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def _build_message(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = cc
    if bcc:
        msg["Bcc"] = bcc
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(body)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    payload: Dict[str, Any] = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id
    return payload


def _extract_plain_text(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
    parts = payload.get("parts") or []
    for part in parts:
        text = _extract_plain_text(part)
        if text:
            return text
    return ""

def _normalize_keywords(raw: Any) -> Dict[str, int]:
    if isinstance(raw, dict):
        return {str(k).lower(): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {str(k).lower(): 1 for k in raw}
    return {}


def _score_text(text: str, keywords: Dict[str, int]) -> int:
    if not text or not keywords:
        return 0
    lower = text.lower()
    score = 0
    for kw, weight in keywords.items():
        if not kw:
            continue
        count = lower.count(kw)
        if count:
            score += count * max(1, int(weight))
    return score

@catch_errors("GMAIL")
def gmail_setup_instructions() -> Dict[str, Any]:
    paths = _get_paths()
    return {
        "success": True,
        "steps": [
            "Create a Google Cloud project and enable Gmail API.",
            "Create OAuth client credentials (Desktop app).",
            f"Save the client JSON to: {paths['client_secret']}",
            "Run gmail_authenticate() once to generate a token.",
        ],
        "client_secret_path": paths["client_secret"],
        "token_path": paths["token"],
    }


@catch_errors("GMAIL")
def gmail_authenticate(scopes: Optional[List[str]] = None, port: int = 0) -> Dict[str, Any]:
    check = _require_google_api()
    if not check["success"]:
        return check
    paths = _get_paths()
    client_secret_path = paths["client_secret"]
    if not os.path.exists(client_secret_path):
        raise ToolError(
            f"Gmail client secret not found at {client_secret_path}",
            code="NO_CLIENT_SECRET",
            context={"client_secret_path": client_secret_path},
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, _get_scopes(scopes))
    creds = flow.run_local_server(port=port)
    os.makedirs(os.path.dirname(paths["token"]), exist_ok=True)
    with open(paths["token"], "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return {"success": True, "token_path": paths["token"]}


@catch_errors("GMAIL")
def gmail_auth_status(scopes: Optional[List[str]] = None) -> Dict[str, Any]:
    paths = _get_paths()
    token_path = paths["token"]
    if not os.path.exists(token_path):
        return {"success": False, "error": "token_missing", "token_path": token_path}
    try:
        creds = _load_credentials(scopes)
    except ToolError as exc:
        return {"success": False, "error": str(exc), "token_path": token_path}
    return {
        "success": True,
        "token_path": token_path,
        "scopes": creds.scopes,
        "valid": creds.valid,
    }


@catch_errors("GMAIL")
def gmail_list_labels() -> Dict[str, Any]:
    creds = _load_credentials()
    service = _build_service(creds)
    resp = service.users().labels().list(userId="me").execute()
    labels = resp.get("labels", [])
    return {"success": True, "labels": labels}


@catch_errors("GMAIL")
def gmail_list_messages(query: str = "", max_results: int = 10, label_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    creds = _load_credentials()
    service = _build_service(creds)
    req = service.users().messages().list(
        userId="me",
        q=query or None,
        maxResults=max(1, min(int(max_results), 100)),
        labelIds=label_ids or None,
    )
    resp = req.execute()
    messages = resp.get("messages", [])
    return {"success": True, "messages": messages, "result_size": resp.get("resultSizeEstimate", 0)}


@catch_errors("GMAIL")
def gmail_get_message(message_id: str, format: str = "full") -> Dict[str, Any]:
    creds = _load_credentials()
    service = _build_service(creds)
    msg = service.users().messages().get(userId="me", id=message_id, format=format).execute()
    payload = msg.get("payload") or {}
    plain = _extract_plain_text(payload) if format in ("full", "metadata") else ""
    return {"success": True, "message": msg, "text": plain}


@catch_errors("GMAIL")
def gmail_set_triage_rules(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Save triage rules. Each rule supports:
    - name (str)
    - query (Gmail search query)
    - add_labels (list[str])
    - remove_labels (list[str])
    - mark_read (bool)
    - draft_reply (dict): {subject, body, reply_to_sender: bool}
    """
    paths = _get_paths()
    _save_json_file(paths["triage_rules"], rules or [])
    return {"success": True, "rules_saved": len(rules or [])}


@catch_errors("GMAIL")
def gmail_get_triage_rules() -> Dict[str, Any]:
    paths = _get_paths()
    rules = _load_json_file(paths["triage_rules"], [])
    return {"success": True, "rules": rules}


@catch_errors("GMAIL")
def gmail_apply_triage_rules(max_per_rule: int = 50, dry_run: bool = False) -> Dict[str, Any]:
    """
    Apply saved triage rules:
    - label messages
    - optionally mark read
    - optionally create reply drafts (never sends)
    """
    creds = _load_credentials()
    service = _build_service(creds)
    paths = _get_paths()
    rules = _load_json_file(paths["triage_rules"], [])
    if not rules:
        return {"success": False, "error": "no_rules"}

    summary = {"rules": [], "dry_run": bool(dry_run)}

    for rule in rules:
        name = rule.get("name") or rule.get("query") or "rule"
        query = rule.get("query", "")
        add_labels = rule.get("add_labels") or []
        remove_labels = rule.get("remove_labels") or []
        if rule.get("mark_read"):
            remove_labels = list(set(remove_labels + ["UNREAD"]))

        score_keywords = _normalize_keywords(rule.get("score_keywords"))
        score_threshold = int(rule.get("score_threshold") or 1)
        score_labels = rule.get("score_labels") or []

        label_map = _ensure_labels(service, add_labels + remove_labels)
        add_ids = [label_map.get(l) for l in add_labels if label_map.get(l)]
        remove_ids = [label_map.get(l) for l in remove_labels if label_map.get(l)]

        resp = service.users().messages().list(
            userId="me",
            q=query or None,
            maxResults=max(1, min(int(max_per_rule), 200)),
        ).execute()
        messages = resp.get("messages", []) or []

        rule_result = {"name": name, "matched": len(messages), "modified": 0, "drafts": 0, "scored": 0}

        for item in messages:
            msg_id = item.get("id")
            if not msg_id:
                continue
            if score_keywords:
                meta = service.users().messages().get(
                    userId="me",
                    id=msg_id,
                    format="metadata",
                    metadataHeaders=["From", "Subject"],
                ).execute()
                payload = meta.get("payload") or {}
                snippet = meta.get("snippet") or ""
                subject = _get_header(payload, "Subject")
                sender = _get_header(payload, "From")
                text = f"{subject}\n{sender}\n{snippet}"
                score = _score_text(text, score_keywords)
                if score < score_threshold:
                    continue
                rule_result["scored"] += 1
                if score_labels:
                    label_map = _ensure_labels(service, score_labels)
                    add_ids = list(set(add_ids + [label_map.get(l) for l in score_labels if label_map.get(l)]))
            if not dry_run:
                if add_ids or remove_ids:
                    body = {"addLabelIds": add_ids, "removeLabelIds": remove_ids}
                    service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
                    rule_result["modified"] += 1

                # Draft reply workflow (never send)
                draft_cfg = rule.get("draft_reply") or {}
                if draft_cfg and draft_cfg.get("body"):
                    meta = service.users().messages().get(
                        userId="me",
                        id=msg_id,
                        format="metadata",
                        metadataHeaders=["From", "Subject"],
                    ).execute()
                    payload = meta.get("payload") or {}
                    sender = _get_header(payload, "From")
                    _, sender_email = parseaddr(sender)
                    if sender_email:
                        subject = draft_cfg.get("subject") or _get_header(payload, "Subject")
                        if subject and not subject.lower().startswith("re:"):
                            subject = f"Re: {subject}"
                        draft = service.users().drafts().create(
                            userId="me",
                            body={"message": _build_message(
                                to=sender_email,
                                subject=subject or "Re:",
                                body=str(draft_cfg.get("body", "")),
                                thread_id=meta.get("threadId"),
                            )},
                        ).execute()
                        if draft.get("id"):
                            rule_result["drafts"] += 1

        summary["rules"].append(rule_result)

    return {"success": True, "summary": summary}


@catch_errors("GMAIL")
def gmail_send_message(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    creds = _load_credentials()
    service = _build_service(creds)
    message = _build_message(to, subject, body, cc=cc, bcc=bcc, reply_to=reply_to, thread_id=thread_id)
    sent = service.users().messages().send(userId="me", body=message).execute()
    return {"success": True, "message_id": sent.get("id"), "thread_id": sent.get("threadId")}


@catch_errors("GMAIL")
def gmail_create_draft(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    creds = _load_credentials()
    service = _build_service(creds)
    message = _build_message(to, subject, body, cc=cc, bcc=bcc, reply_to=reply_to, thread_id=thread_id)
    draft = service.users().drafts().create(userId="me", body={"message": message}).execute()
    return {"success": True, "draft_id": draft.get("id")}


@catch_errors("GMAIL")
def gmail_list_outbox() -> Dict[str, Any]:
    paths = _get_paths()
    outbox = _load_json_file(paths["outbox"], [])
    return {"success": True, "outbox": outbox}


@catch_errors("GMAIL")
def gmail_schedule_send(
    to: str,
    subject: str,
    body: str,
    send_after_iso: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
    thread_id: Optional[str] = None,
    approved: bool = False,
) -> Dict[str, Any]:
    """
    Queue a scheduled email. It will only send when:
    - current time >= send_after_iso
    - approved == True (explicit gate)
    """
    paths = _get_paths()
    outbox = _load_json_file(paths["outbox"], [])
    item = {
        "id": f"outbox_{len(outbox) + 1}",
        "to": to,
        "subject": subject,
        "body": body,
        "cc": cc,
        "bcc": bcc,
        "reply_to": reply_to,
        "thread_id": thread_id,
        "send_after": send_after_iso,
        "approved": bool(approved),
        "status": "queued",
    }
    outbox.append(item)
    _save_json_file(paths["outbox"], outbox)
    return {"success": True, "queued": item}


@catch_errors("GMAIL")
def gmail_approve_send(item_id: str, approved: bool = True) -> Dict[str, Any]:
    paths = _get_paths()
    outbox = _load_json_file(paths["outbox"], [])
    updated = False
    for item in outbox:
        if item.get("id") == item_id:
            item["approved"] = bool(approved)
            updated = True
            break
    if updated:
        _save_json_file(paths["outbox"], outbox)
        return {"success": True, "item_id": item_id, "approved": bool(approved)}
    return {"success": False, "error": "item_not_found", "item_id": item_id}


@catch_errors("GMAIL")
def gmail_approve_due(send_before_iso: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Approve queued items whose send_after <= send_before_iso (or now if omitted).
    """
    from datetime import datetime, timezone

    paths = _get_paths()
    outbox = _load_json_file(paths["outbox"], [])
    if not outbox:
        return {"success": True, "approved": 0}

    if send_before_iso:
        cutoff = datetime.fromisoformat(send_before_iso)
    else:
        cutoff = datetime.now(timezone.utc)

    approved = 0
    for item in outbox:
        if approved >= max(1, int(limit)):
            break
        if item.get("status") != "queued":
            continue
        try:
            send_time = datetime.fromisoformat(item.get("send_after"))
        except Exception:
            send_time = cutoff
        if send_time <= cutoff and not item.get("approved"):
            item["approved"] = True
            approved += 1

    _save_json_file(paths["outbox"], outbox)
    return {"success": True, "approved": approved}

@catch_errors("GMAIL")
def gmail_run_outbox(now_iso: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Send approved, due messages from the outbox.
    """
    from datetime import datetime, timezone

    paths = _get_paths()
    outbox = _load_json_file(paths["outbox"], [])
    if not outbox:
        return {"success": True, "sent": 0}

    if now_iso:
        now = datetime.fromisoformat(now_iso)
    else:
        now = datetime.now(timezone.utc)

    creds = _load_credentials()
    service = _build_service(creds)

    sent = 0
    for item in outbox:
        if sent >= max(1, int(limit)):
            break
        if item.get("status") != "queued":
            continue
        if not item.get("approved"):
            continue
        send_after = item.get("send_after")
        try:
            send_time = datetime.fromisoformat(send_after)
        except Exception:
            send_time = now
        if send_time > now:
            continue

        message = _build_message(
            to=item.get("to", ""),
            subject=item.get("subject", ""),
            body=item.get("body", ""),
            cc=item.get("cc"),
            bcc=item.get("bcc"),
            reply_to=item.get("reply_to"),
            thread_id=item.get("thread_id"),
        )
        sent_msg = service.users().messages().send(userId="me", body=message).execute()
        item["status"] = "sent"
        item["message_id"] = sent_msg.get("id")
        sent += 1

    _save_json_file(paths["outbox"], outbox)
    return {"success": True, "sent": sent}


@catch_errors("GMAIL")
def gmail_modify_labels(message_id: str, add_labels: Optional[List[str]] = None, remove_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    creds = _load_credentials()
    service = _build_service(creds)
    body = {
        "addLabelIds": add_labels or [],
        "removeLabelIds": remove_labels or [],
    }
    resp = service.users().messages().modify(userId="me", id=message_id, body=body).execute()
    return {"success": True, "message": resp}
