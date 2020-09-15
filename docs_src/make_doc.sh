#!/bin/bash

pandoc --defaults=./pandoc.yaml \
    ./DIC_introduction.md -o ../docs/index.html \
    --metadata title="introduction"

pandoc --defaults=./pandoc.yaml \
    ./DIC_softwares.md -o ../docs/list_DICsoftwares.html \
    --metadata title="list of DIC softwares"

pandoc --defaults=./pandoc.yaml \
    ./images_registration.md -o ../docs/images_registration.html \
    --metadata title="math of images registration"

pandoc --defaults=./pandoc.yaml \
    ./list_of_references.md -o ../docs/list_of_references.html \
    --metadata title="list of references"


pdoc --html --output-dir ../docs ../stretchablecorr --force


