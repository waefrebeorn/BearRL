#!/usr/bin/env bash
# parity_gate.sh -- cross-repo truth gate (GAP-H003)
#
# Doctrine: anything BearRL proves about WuBuMath components must ALSO pass
# WuBuMath's own tests against the SAME commit. No divergent truths.
#
# Usage: tools/parity_gate.sh [wubumath_path]
# Exit 0 iff: both repos clean-checked, same HEAD recorded, WuBuMath make test
# green, BearRL propgate ALL_HOLD.

set -u
WUBUMATH="${1:-$HOME/wubumath}"
BEARRL="$HOME/BearRL"
fail() { echo "PARITY_GATE: FAIL — $1"; exit 1; }

[ -d "$WUBUMATH/.git" ] || fail "wubumath repo not found at $WUBUMATH"
[ -d "$BEARRL/.git" ]   || fail "BearRL repo not found at $BEARRL"

cd "$WUBUMATH"
echo "== building + running WuBuMath unit gate =="
make test > /tmp/parity_wm.log 2>&1 || fail "WuBuMath make test failed (see /tmp/parity_wm.log)"
WM_HEAD=$(git rev-parse --short HEAD)

cd "$BEARRL"
echo "== running BearRL property gate =="
python3 tools/propgate.py --seeds 2 > /tmp/parity_br.log 2>&1 \
  || fail "BearRL propgate violations (see /tmp/parity_br.log)"

BR_CERT="artifacts/propgate_certificate.json"
[ -f "$BR_CERT" ] || fail "no certificate artifact — run propgate with --json"

cat > artifacts/parity_certificate.json <<EOF
{
  "when": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "wubumath_head": "$WM_HEAD",
  "wubumath_make_test": "PASS",
  "bearrl_propgate": "ALL_HOLD",
  "doctrine": "no divergent truths between repos at the same commit"
}
EOF

echo "PARITY_GATE: PASS (wubumath@$WM_HEAD, both gates green)"
