#!/bin/bash

pandoc --defaults=./docs_src/pandoc.yaml \
    ./docs_src/DIC_introduction.md -o ./docs/index.html \
    --metadata title="introduction"


pandoc --defaults=./docs_src/pandoc.yaml \
    ./docs_src/DIC_softwares.md -o ./docs/list_DICsoftwares.html \
    --metadata title="list of DIC softwares"

pandoc --defaults=./docs_src/pandoc.yaml \
    ./docs_src/images_registration.md -o ./docs/images_registration.html \
    --metadata title="math of images registration"

pandoc --defaults=./docs_src/pandoc.yaml \
    ./docs_src/math_DIC.md -o ./docs/math_DIC.html \
    --metadata title="math DIC"


pdoc --html --output-dir docs stretchablecorr --force