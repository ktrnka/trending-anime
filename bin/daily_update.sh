#!/bin/sh
set -x

rm -rf site
mkdir -p site
python make_old_site.py -v config 2015/04/01 site/winter2015.html res/extra-fonts.css res/fonts
python make_old_site.py -v config 2015/07/01 site/spring2015.html res/extra-fonts.css res/fonts
python make_old_site.py -v config 2015/10/01 site/summer2015.html res/extra-fonts.css res/fonts
python make_site.py -v config site/index.html res/extra-fonts.css res/fonts res/about.html
python deploy_site.py -v config site