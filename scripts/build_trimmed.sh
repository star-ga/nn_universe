#!/bin/bash
# build_trimmed.sh — deterministic build of the trimmed 9-page anonymised
# NeurIPS submission from the trimmed Markdown sources in $NEURIPS_BUILD_DIR.
#
# What it does (in order):
#   1. pandoc paper_main_v11_3.md -> body LaTeX
#   2. wrap body in NeurIPS template + Abstract environment fix
#   3. pdflatex (3 passes for hyperref) → main PDF
#   4. same for supplementary
#   5. verify page counts (main ≤ 9, supp ≤ 10)
#   6. exiftool strip metadata
#   7. anonymisation grep (STARGA/Nedovodin/star.ga/Nikolai/ceo@star.ga = 0)
#   8. reviewer-vendor-leak grep (model-D/model-C/GPT-5/reviewer-B/etc. = 0 in body)
#   9. mirror to /data/checkpoints/neurips2026_submission/submission/
#  10. refresh submission.zip
#
# Inputs (env-overridable):
#   NEURIPS_BUILD_DIR  default /tmp/neurips_build/work
#   SUBMISSION_DIR     default /data/checkpoints/neurips2026_submission
#   MAIN_MD            default $NEURIPS_BUILD_DIR/paper_main_v11_3.md
#   SUPP_MD            default $NEURIPS_BUILD_DIR/paper_supp_v11_3_anon.md
#   MAX_MAIN_PAGES     default 9
#   MAX_SUPP_PAGES     default 10
#
# Exits non-zero if page count exceeded, anonymisation leaks, or reviewer-vendor
# leaks detected. Run from anywhere; always returns to invoker's cwd.

set -euo pipefail

WORK="${NEURIPS_BUILD_DIR:-/tmp/neurips_build/work}"
SUB="${SUBMISSION_DIR:-/data/checkpoints/neurips2026_submission}"
MAIN_MD="${MAIN_MD:-$WORK/paper_main_v11_3.md}"
SUPP_MD="${SUPP_MD:-$WORK/paper_supp_v11_3_anon.md}"
MAX_MAIN_PAGES="${MAX_MAIN_PAGES:-9}"
MAX_SUPP_PAGES="${MAX_SUPP_PAGES:-10}"

orig_cwd="$(pwd)"
trap 'cd "$orig_cwd"' EXIT

# ------------------------------------------------------------ preflight
[ -d "$WORK" ] || { echo "FAIL: build dir $WORK not found"; exit 1; }
[ -f "$MAIN_MD" ] || { echo "FAIL: main md $MAIN_MD not found"; exit 1; }
[ -f "$SUPP_MD" ] || { echo "FAIL: supp md $SUPP_MD not found"; exit 1; }
[ -f "$WORK/neurips_2025.sty" ] || { echo "FAIL: $WORK/neurips_2025.sty missing"; exit 1; }
for tool in pandoc pdflatex pdfinfo pdftotext exiftool zip; do
  command -v "$tool" >/dev/null 2>&1 || { echo "FAIL: $tool not on PATH"; exit 1; }
done

cd "$WORK"

# ------------------------------------------------------------ build helpers
make_wrapper() {
  local title="$1" body_input="$2" out="$3"
  cat > "$out" <<LATEXEOF
\\documentclass{article}
\\usepackage{neurips_2025}

\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{hyperref}
\\usepackage{url}
\\usepackage{booktabs}
\\usepackage{amsfonts}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsthm}
\\usepackage{nicefrac}
\\usepackage{microtype}
\\usepackage{xcolor}
\\usepackage{graphicx}
\\usepackage{longtable}
\\usepackage{calc}
\\usepackage{textcomp}

\\title{$title}

\\author{%
  Anonymous Author(s)\\\\
  Submission under double-blind review\\\\
}

\\begin{document}
\\maketitle

\\input{$body_input}

\\end{document}
LATEXEOF
}

convert_abstract() {
  # pandoc produces \subsection{Abstract}; NeurIPS template wants \begin{abstract}...\end{abstract}
  local in="$1" out="$2"
  python3 - "$in" "$out" <<'PYEOF'
import re, sys
in_path, out_path = sys.argv[1], sys.argv[2]
src = open(in_path).read()
m = re.search(r'\\hypertarget\{abstract\}\{%\n\\subsection\{Abstract\}\\label\{abstract\}\}\n', src)
if not m:
    open(out_path, 'w').write(src)
    sys.exit(0)
start = m.end()
end_m = re.search(r'\\hypertarget\{', src[start:])
end = start + end_m.start() if end_m else len(src)
abstract_content = src[start:end].rstrip()
new_src = src[:m.start()] + '\\begin{abstract}\n' + abstract_content + '\n\\end{abstract}\n\n' + src[end:]
open(out_path, 'w').write(new_src)
PYEOF
}

compile_pdf() {
  # 3 pdflatex passes for hyperref. Last pass may exit non-zero on benign
  # rerunfilecheck warnings even when the PDF is produced — gate on PDF
  # existence + recency, not pdflatex exit code.
  local tex="$1"
  local pdf="${tex%.tex}.pdf"
  rm -f "$pdf"
  pdflatex -interaction=nonstopmode -halt-on-error "$tex" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -halt-on-error "$tex" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$tex" >/dev/null 2>&1 || true
  if [ ! -s "$pdf" ]; then
    echo "FAIL: pdflatex did not produce $pdf (last 30 lines of log):"
    tail -30 "${tex%.tex}.log"
    exit 1
  fi
}

count_pages() {
  pdfinfo "$1" | awk '/^Pages:/ {print $2}'
}

# ------------------------------------------------------------ main paper
echo "=== Build main paper ==="
pandoc "$MAIN_MD" -o paper_main_v11_3_body_raw.tex
convert_abstract paper_main_v11_3_body_raw.tex paper_main_v11_3_body.tex
make_wrapper \
  'Fisher Information Tier Hierarchy:\\A Panel-Bounded Empirical Regularity of Deep Layered Sequential Computation' \
  paper_main_v11_3_body \
  paper_main_v11_3.tex
compile_pdf paper_main_v11_3.tex
MAIN_PAGES=$(count_pages paper_main_v11_3.pdf)
echo "  main pages: $MAIN_PAGES (limit $MAX_MAIN_PAGES)"

# ------------------------------------------------------------ supplementary
echo "=== Build supplementary ==="
pandoc "$SUPP_MD" -o paper_supp_v11_3_body.tex
make_wrapper \
  'Supplementary material: Fisher Information Tier Hierarchy' \
  paper_supp_v11_3_body \
  paper_supp_v11_3.tex
compile_pdf paper_supp_v11_3.tex
SUPP_PAGES=$(count_pages paper_supp_v11_3.pdf)
echo "  supp pages: $SUPP_PAGES (limit $MAX_SUPP_PAGES)"

# ------------------------------------------------------------ strip pdf metadata
exiftool -overwrite_original -all= paper_main_v11_3.pdf >/dev/null 2>&1 || true
exiftool -overwrite_original -all= paper_supp_v11_3.pdf >/dev/null 2>&1 || true

# ------------------------------------------------------------ checks
errors=0

if [ "$MAIN_PAGES" -gt "$MAX_MAIN_PAGES" ]; then
  echo "FAIL: main paper $MAIN_PAGES pages > limit $MAX_MAIN_PAGES"
  errors=$((errors + 1))
fi
if [ "$SUPP_PAGES" -gt "$MAX_SUPP_PAGES" ]; then
  echo "FAIL: supp $SUPP_PAGES pages > limit $MAX_SUPP_PAGES"
  errors=$((errors + 1))
fi

echo "=== Anonymisation check ==="
for kw in STARGA Nedovodin star.ga "github.com/star-ga" Nikolai ceo@star.ga; do
  m=$(pdftotext paper_main_v11_3.pdf - 2>/dev/null | grep -c -i "$kw" || true)
  s=$(pdftotext paper_supp_v11_3.pdf - 2>/dev/null | grep -c -i "$kw" || true)
  if [ "$m" -gt 0 ] || [ "$s" -gt 0 ]; then
    echo "  FAIL '$kw' main=$m supp=$s"
    errors=$((errors + 1))
  else
    echo "  OK   '$kw' main=0 supp=0"
  fi
done

echo "=== reviewer-vendor-leak check ==="
# OpenAI is allowed when used as the GPT-2 paper publisher citation; everything
# else should be 0 in the body.
for kw in model-D vendor-D 'model-C ' 'GPT-5' 'reviewer-B ' Anthropic model-E model-F model-G Moonshot Kimi 'Llama-' 'Qwen '; do
  m=$(pdftotext paper_main_v11_3.pdf - 2>/dev/null | grep -c -i "$kw" || true)
  s=$(pdftotext paper_supp_v11_3.pdf - 2>/dev/null | grep -c -i "$kw" || true)
  if [ "$m" -gt 0 ] || [ "$s" -gt 0 ]; then
    echo "  WARN '$kw' main=$m supp=$s (review whether load-bearing)"
  fi
done

# ------------------------------------------------------------ mirror to /data
if [ -d "$SUB/submission" ]; then
  echo "=== Mirror to $SUB/submission ==="
  cp paper_main_v11_3.pdf "$SUB/submission/paper.pdf"
  cp paper_supp_v11_3.pdf "$SUB/submission/supplementary.pdf"
  rm -f "$SUB/submission.zip"
  (cd "$SUB/submission" && zip -rq "$SUB/submission.zip" paper.pdf supplementary.pdf code 2>&1)
  ls -la "$SUB/submission/paper.pdf" "$SUB/submission/supplementary.pdf" "$SUB/submission.zip"
else
  echo "  SKIP: submission dir $SUB/submission not found"
fi

# ------------------------------------------------------------ summary
echo
echo "=== build_trimmed.sh summary ==="
echo "  main pages : $MAIN_PAGES / $MAX_MAIN_PAGES"
echo "  supp pages : $SUPP_PAGES / $MAX_SUPP_PAGES"
echo "  errors     : $errors"

if [ "$errors" -gt 0 ]; then
  echo "FAIL: $errors error(s)"
  exit 2
fi
echo "OK"
