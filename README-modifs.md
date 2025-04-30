# BabelDOC + some fixes for Japanese translation

## Better translation of multicolumns 

BabelDOC translates per text block.  If a document uses 2 column layout and a sentence wraps over columns, it is splitted into 2 blocks and they are translated separately, which causes mistranslation.

A hack is introduced to translate such a sentence seemingly splitted by columns in one piece.

## Japanese translation

LLM prompt tweaks for Japanese translation

## Font selection

### Workaround of selecting serif/sans-serif fonts

`--force-font serif/sans-serif` option is added to workarond a bug of PyMuPDF's serif detection.

### Fixes around Japanese font handling

BIZ UD fonts for Japanese rendering:

- BIZUDGothic-Regular.ttf
- BIZUDGothic-Bold.ttf
- BIZUDMincho-Regular.ttf
- BIZUDMincho-Bold.ttf

For non CJK fonts, the following fonts are used: 

- NotoSans-Regular.ttf
- NotoSans-Bold.ttf

They are automatically downloaded.

## Font subsetting workaround

It seems PyMuPDF's font subsetting has a bug and some glyphs are lost during the subsetting.  A patch is introduced to fallback to a safer algorithm using `fonttools`.
