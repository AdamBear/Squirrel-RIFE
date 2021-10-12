# caution: save and load ts for several times before pylupdate, to fix UTF-8 gibberish
# sub-module need retranslate as well
# translate does not support long text or multi arguments in methods or ... change it to '%s' and use ''.format
CODECFORTR = UTF-8
CODECFORSRC = UTF-8
SOURCES += SVFI_UI.py \
           SVFI_help.py \
           SVFI_about.py \
           SVFI_preference.py \
           SVFI_preview_args.py \
           SVFI_input_item.py \
           RIFE_GUI_Backend.py

TRANSLATIONS += SVFI_UI.zh.ts \
                SVFI_UI.en.ts
