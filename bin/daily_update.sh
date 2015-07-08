#!/bin/sh
python make_old_site.py -v config 2015/04/01 bitballoon/winter2015.html res/extra-fonts.css res/fonts res/about.html
python make_old_site.py -v config 2015/07/01 bitballoon/spring2015.html res/extra-fonts.css res/fonts res/about.html
python make_site.py -v config bitballoon res/extra-fonts.css res/fonts res/about.html