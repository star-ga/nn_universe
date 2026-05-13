# Build pipeline for the NeurIPS submission PDF.
#
#   make pdf      → docs/build/fim_tier_hierarchy_neurips2026.pdf
#   make clean    → wipe intermediate LaTeX files (keep .sty + .tex source)
#
# Inputs:
#   ../fim_tier_hierarchy_neurips2026.md   (paper source, markdown)
#   neurips_2025.sty                        (NeurIPS template style file)
#   wrapper.tex                              (LaTeX preamble + \input{body.tex})
#
# Toolchain required: pandoc, pdflatex.

SRC    = ../fim_tier_hierarchy_neurips2026.md
BODY   = body.tex
PDF    = fim_tier_hierarchy_neurips2026.pdf
JOB    = wrapper

.PHONY: pdf clean

pdf: $(PDF)

# Strip the H1 title and metadata block from the markdown source before
# converting — the LaTeX wrapper handles \title{} and \maketitle.
$(BODY): $(SRC)
	@echo "  pandoc  $< -> $@"
	sed -n '/^---$$/,$$p' $< | sed '1d' > _body_stripped.md
	pandoc _body_stripped.md \
	    --from=markdown \
	    --to=latex \
	    --output=$@ \
	    --listings \
	    --top-level-division=section
	rm -f _body_stripped.md

$(PDF): wrapper.tex $(BODY) neurips_2025.sty
	@echo "  pdflatex (pass 1)"
	xelatex -interaction=nonstopmode -halt-on-error $(JOB).tex > /tmp/pdflatex.log 2>&1 \
	    || (tail -50 /tmp/pdflatex.log && false)
	@echo "  pdflatex (pass 2 for cross-refs)"
	xelatex -interaction=nonstopmode -halt-on-error $(JOB).tex > /tmp/pdflatex.log 2>&1 \
	    || (tail -50 /tmp/pdflatex.log && false)
	mv $(JOB).pdf $(PDF)
	@echo "  ✓ built $(PDF)"

clean:
	rm -f body.tex _body_stripped.md $(JOB).aux $(JOB).log $(JOB).out $(JOB).pdf
