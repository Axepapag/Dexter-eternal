#!/usr/bin/env python3
"""
Business Operations for Dexter
Income tracking, affiliate management, and revenue intelligence
Optimized for Jeffrey Gliksman's $10K/month goal
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Dexter data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

REVENUE_FILE = DATA_DIR / "revenue_ledger.json"
AFFILIATE_FILE = DATA_DIR / "affiliate_programs.json"
BUSINESS_DB = DATA_DIR / "business.db"

__tool_prefix__ = "business"

# --- Default Affiliate Programs ---
DEFAULT_AFFILIATES = {
    "amazon": {
        "status": "active",
        "commission": "1-10%",
        "payout": "30 days",
        "tag": "gliksbot-20",
        "program": "Amazon Associates",
        "niche": "General",
        "estimated_monthly": 500
    },
    "shareasale": {
        "status": "research",
        "commission": "5-50%",
        "signup_url": "https://www.shareasale.com/info/affiliate-signup",
        "program": "ShareASale",
        "niche": "Various",
        "estimated_monthly": 0
    },
    "cj_affiliate": {
        "status": "research",
        "commission": "5-25%",
        "signup_url": "https://www.cj.com/publisher-sign-up",
        "program": "Commission Junction",
        "niche": "Enterprise",
        "estimated_monthly": 0
    },
    "learnerithm": {
        "status": "recommended",
        "commission": "60% recurring",
        "payout": "Monthly",
        "program": "Learnerithm AI",
        "niche": "AI/Education",
        "estimated_monthly": 600,
        "url": "https://partner.learnrithm.com"
    },
    "adcreative": {
        "status": "recommended",
        "commission": "40% recurring",
        "payout": "Monthly",
        "program": "AdCreative.ai",
        "niche": "Marketing",
        "estimated_monthly": 400,
        "url": "https://www.adcreative.ai/affiliate"
    },
    "gohighlevel": {
        "status": "recommended",
        "commission": "40% recurring",
        "payout": "Monthly",
        "program": "GoHighLevel",
        "niche": "Marketing/CRM",
        "estimated_monthly": 450,
        "url": "https://www.gohighlevel.com/affiliate"
    }
}

# --- Database Setup ---
def _init_business_db():
    """Initialize SQLite business database"""
    conn = sqlite3.connect(str(BUSINESS_DB))
    cursor = conn.cursor()
    
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS income_missions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        category TEXT,
        target_amount REAL DEFAULT 10000.0,
        current_amount REAL DEFAULT 0.0,
        deadline TEXT,
        priority INTEGER DEFAULT 1,
        status TEXT DEFAULT 'active',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        completed_at TEXT
    );
    
    CREATE TABLE IF NOT EXISTS revenue_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        amount REAL NOT NULL,
        currency TEXT DEFAULT 'USD',
        category TEXT,
        notes TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS affiliate_commissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        program TEXT NOT NULL,
        commission_amount REAL DEFAULT 0.0,
        referred_user TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        paid BOOLEAN DEFAULT 0
    );
    """)
    
    # Add default mission if not exists
    cursor.execute("SELECT COUNT(*) FROM income_missions WHERE title = 'Monthly Income Goal'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO income_missions (title, description, target_amount, category, priority)
        VALUES ('Monthly Income Goal', 'Achieve $10,000/month from affiliate marketing', 10000.0, 'affiliate', 1)
        """)
    
    conn.commit()
    conn.close()

# Initialize on import
_init_business_db()

# --- Tools ---

def business_get_affiliates() -> Dict[str, Any]:
    """
    Get all tracked affiliate programs with status and potential earnings.
    
    Returns affiliate programs, their status, and estimated monthly earnings.
    """
    if not AFFILIATE_FILE.exists():
        with open(AFFILIATE_FILE, 'w') as f:
            json.dump(DEFAULT_AFFILIATES, f, indent=2)
        return {
            "success": True,
            "programs": DEFAULT_AFFILIATES,
            "total_programs": len(DEFAULT_AFFILIATES),
            "active_programs": sum(1 for p in DEFAULT_AFFILIATES.values() if p.get('status') == 'active'),
            "potential_monthly": sum(p.get('estimated_monthly', 0) for p in DEFAULT_AFFILIATES.values())
        }
    
    with open(AFFILIATE_FILE, 'r') as f:
        programs = json.load(f)
    
    return {
        "success": True,
        "programs": programs,
        "total_programs": len(programs),
        "active_programs": sum(1 for p in programs.values() if p.get('status') == 'active'),
        "potential_monthly": sum(p.get('estimated_monthly', 0) for p in programs.values())
    }

def business_add_affiliate(name: str, program_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add or update an affiliate program.
    
    Args:
        name: Program identifier (e.g., "amazon", "learnerithm")
        program_data: Dict with status, commission, url, etc.
    """
    affiliates = business_get_affiliates().get("programs", {})
    affiliates[name] = program_data
    
    with open(AFFILIATE_FILE, 'w') as f:
        json.dump(affiliates, f, indent=2)
    
    return {
        "success": True,
        "message": f"Added/updated {name} affiliate program",
        "program": program_data
    }

def business_add_revenue(source: str, amount: float, currency: str = "USD", 
                         category: str = "general", notes: str = "") -> Dict[str, Any]:
    """
    Record a revenue event to both JSON ledger and SQLite database.
    
    Args:
        source: Revenue source (e.g., "Amazon Associates", "Fiverr")
        amount: Revenue amount
        currency: Currency code (default USD)
        category: Revenue category (affiliate, freelance, product, etc.)
        notes: Additional notes
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "amount": amount,
        "currency": currency,
        "category": category,
        "notes": notes
    }
    
    # Update JSON ledger
    ledger = []
    if REVENUE_FILE.exists():
        with open(REVENUE_FILE, 'r') as f:
            try:
                ledger = json.load(f)
            except json.JSONDecodeError:
                ledger = []
    
    ledger.append(entry)
    
    with open(REVENUE_FILE, 'w') as f:
        json.dump(ledger, f, indent=2)
    
    # Update SQLite
    conn = sqlite3.connect(str(BUSINESS_DB))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO revenue_entries (source, amount, currency, category, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (source, amount, currency, category, notes))
    conn.commit()
    conn.close()
    
    # Progress toward goal
    progress = business_get_progress()
    
    return {
        "success": True, 
        "message": f"Recorded ${amount:.2f} from {source}",
        "total_entries": len(ledger),
        "monthly_progress": progress.get("current_month_percent", 0),
        "remaining_to_goal": progress.get("remaining", 10000)
    }

def business_get_total_revenue(days: int = 30) -> Dict[str, Any]:
    """
    Calculate total revenue for specified time period.
    
    Args:
        days: Number of days to look back (default 30)
    """
    conn = sqlite3.connect(str(BUSINESS_DB))
    cursor = conn.cursor()
    
    since_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    cursor.execute("""
        SELECT SUM(amount), COUNT(*), AVG(amount)
        FROM revenue_entries
        WHERE timestamp > ?
    """, (since_date,))
    
    result = cursor.fetchone()
    conn.close()
    
    total = result[0] or 0.0
    count = result[1] or 0
    avg = result[2] or 0.0
    
    return {
        "success": True,
        "total": round(total, 2),
        "currency": "USD",
        "count": count,
        "average": round(avg, 2),
        "period_days": days,
        "daily_average": round(total / max(days, 1), 2)
    }

def business_get_progress() -> Dict[str, Any]:
    """
    Get progress toward $10,000/month income goal.
    """
    # Get current month's revenue
    current_month = business_get_total_revenue(days=30)
    current_amount = current_month.get("total", 0.0)
    
    GOAL = 10000.0
    remaining = max(0, GOAL - current_amount)
    percent = min(100, (current_amount / GOAL) * 100)
    
    # Days remaining in month
    today = datetime.now()
    if today.month == 12:
        last_day = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        last_day = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    days_remaining = (last_day - today).days
    
    # Projection
    daily_avg = current_month.get("daily_average", 0)
    projected_month = daily_avg * 30
    
    return {
        "success": True,
        "goal": GOAL,
        "current_month": round(current_amount, 2),
        "remaining": round(remaining, 2),
        "current_month_percent": round(percent, 1),
        "days_remaining_in_month": days_remaining,
        "daily_average_needed": round(remaining / max(days_remaining, 1), 2),
        "projected_month_total": round(projected_month, 2),
        "on_track": projected_month >= GOAL,
        "status": "ahead" if projected_month >= GOAL else "behind" if remaining > (GOAL * 0.5) else "catching_up"
    }

def business_get_top_sources(limit: int = 5) -> Dict[str, Any]:
    """
    Get top revenue sources by amount.
    
    Args:
        limit: Number of top sources to return
    """
    conn = sqlite3.connect(str(BUSINESS_DB))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT source, SUM(amount) as total, COUNT(*) as count
        FROM revenue_entries
        GROUP BY source
        ORDER BY total DESC
        LIMIT ?
    """, (limit,))
    
    sources = [
        {"source": row[0], "total": round(row[1], 2), "transactions": row[2]}
        for row in cursor.fetchall()
    ]
    
    conn.close()
    
    return {
        "success": True,
        "top_sources": sources,
        "total_sources": len(sources)
    }

def business_get_income_report() -> Dict[str, Any]:
    """
    Get comprehensive income report including affiliates, revenue, and progress.
    """
    affiliates = business_get_affiliates()
    revenue = business_get_total_revenue(days=30)
    progress = business_get_progress()
    top_sources = business_get_top_sources()
    
    return {
        "success": True,
        "report_date": datetime.now().isoformat(),
        "summary": {
            "monthly_goal": 10000.0,
            "current_month_revenue": revenue.get("total", 0),
            "progress_percent": progress.get("current_month_percent", 0),
            "active_affiliate_programs": affiliates.get("active_programs", 0),
            "potential_from_recommended": sum(
                p.get("estimated_monthly", 0) 
                for p in affiliates.get("programs", {}).values() 
                if p.get("status") == "recommended"
            )
        },
        "details": {
            "revenue": revenue,
            "progress": progress,
            "top_sources": top_sources,
            "affiliates": affiliates
        }
    }

def business_recommend_next_action() -> Dict[str, Any]:
    """
    AI-driven recommendation for next income-generating action based on current status.
    """
    progress = business_get_progress()
    affiliates = business_get_affiliates()
    
    status = progress.get("status", "behind")
    recommendations = []
    
    if status == "behind":
        recommendations.append({
            "priority": "HIGH",
            "action": "Sign up for high-commission affiliate programs",
            "programs": ["learnerithm", "adcreative", "gohighlevel"],
            "reason": f"Currently ${progress.get('remaining', 10000):.0f} behind goal with {progress.get('days_remaining_in_month', 30)} days left"
        })
    
    # Check for research programs that should be activated
    research_programs = [
        name for name, data in affiliates.get("programs", {}).items()
        if data.get("status") == "research"
    ]
    
    if research_programs:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Complete signup for researched programs",
            "programs": research_programs,
            "reason": "Programs researched but not yet activated"
        })
    
    # If on track, recommend scaling
    if progress.get("on_track", False):
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Scale content production to increase earnings",
            "tactics": ["Increase posting frequency", "Test new affiliate products", "Optimize conversion rates"],
            "reason": "On track to meet goal - time to optimize"
        })
    
    return {
        "success": True,
        "status": status,
        "recommendations": recommendations,
        "urgency": "high" if status == "behind" else "medium"
    }

# Backwards compatibility aliases
get_affiliates = business_get_affiliates
add_revenue = business_add_revenue
get_total_revenue = business_get_total_revenue
get_progress = business_get_progress
