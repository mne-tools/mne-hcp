clean:
        rm -rf $(BUILDDIR)/*
        rm -rf auto_examples/
        rm -rf modules/generated/*

html-noplot:
        $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
        @echo
        @echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
