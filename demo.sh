#!/usr/bin/env bash
# demo.sh — Full end-to-end walkthrough of the Academic Research Planner
# Run from the project root with the server already running on :8000
# Usage: bash demo.sh

set -euo pipefail

BASE="http://localhost:8000"
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pp() { python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || cat; }
bold() { printf '\033[1m%s\033[0m\n' "$1"; }
step() { echo; echo "$SEP"; bold "STEP $1 — $2"; echo "$SEP"; }

# ── STEP 0 — Health check ─────────────────────────────────────────────────────
step 0 "Health check"
curl -s "$BASE/health" | pp

# ── STEP 1 — Onboarding: URL ingestion ───────────────────────────────────────
step 1 "Onboard with a real ArXiv paper (Attention Is All You Need)"
ONBOARD=$(curl -s -X POST "$BASE/research-planner/onboarding" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://arxiv.org/pdf/1706.03762"],
    "field_hint": "deep learning, transformer architecture, NLP"
  }')

echo "$ONBOARD" | pp

THREAD_ID=$(echo "$ONBOARD" | python3 -c "import sys,json; print(json.load(sys.stdin)['thread_id'])")
bold "Thread ID: $THREAD_ID"

# ── STEP 2 — Chat: query the corpus ──────────────────────────────────────────
step 2 "Chat — ask the agent to search the corpus"
CHAT1=$(curl -s -X POST "$BASE/research-planner/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"thread_id\": \"$THREAD_ID\",
    \"user_message\": \"Search the corpus and summarise the key contributions of this paper.\",
    \"current_phase\": \"discovery\",
    \"is_deliverable_accepted\": false
  }")

echo "$CHAT1" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Phase:', d.get('current_phase'))
print()
print(d.get('response', '')[:800])
print('...')
"

# ── STEP 3 — Chat: generate discovery deliverable ────────────────────────────
step 3 "Chat — generate the Discovery deliverable"
CHAT2=$(curl -s -X POST "$BASE/research-planner/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"thread_id\": \"$THREAD_ID\",
    \"user_message\": \"My research intent is to build on transformer architectures for multilingual NLP. Target venue: ACL 2025, 8 pages, ACL citation style. Generate the discovery deliverable.\",
    \"current_phase\": \"discovery\",
    \"is_deliverable_accepted\": false
  }")

echo "$CHAT2" | python3 -c "
import sys, json
d = json.load(sys.stdin)
has_del = bool(d.get('deliverables_markdown', {}).get('discovery'))
print('Discovery deliverable generated:', has_del)
print('Phase:', d.get('current_phase'))
print()
print(d.get('response', '')[:600])
print('...')
"

# ── STEP 4 — Accept discovery and advance to clustering ──────────────────────
step 4 "Accept Discovery deliverable → advance to Clustering phase"
CHAT3=$(curl -s -X POST "$BASE/research-planner/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"thread_id\": \"$THREAD_ID\",
    \"user_message\": \"This looks great, I accept the discovery deliverable. Let's move to clustering.\",
    \"current_phase\": \"discovery\",
    \"is_deliverable_accepted\": true
  }")

echo "$CHAT3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('New phase:         ', d.get('current_phase'))
print('Accepted so far:   ', d.get('accepted_deliverables'))
print()
print(d.get('response', '')[:400])
print('...')
"

# ── STEP 5 — GET deliverables ─────────────────────────────────────────────────
step 5 "GET /deliverables — retrieve all structured deliverables"
curl -s "$BASE/research-planner/deliverables/$THREAD_ID" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Thread:           ', d.get('thread_id'))
print('Current phase:    ', d.get('current_phase'))
print('Accepted:         ', d.get('accepted_deliverables'))
print()
disc = d.get('discovery_markdown') or ''
if disc:
    print('--- Discovery Markdown (first 600 chars) ---')
    print(disc[:600])
    print('...')
"

echo
echo "$SEP"
bold "Demo complete. Thread: $THREAD_ID"
echo "$SEP"
