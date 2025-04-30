# BabelDOC + 日本語対応

## 使用方法

オリジナルに準じます:

```shell
% uv run babeldoc \
    -li en -lo jp \
    --openai \
    --openai-api-key OPENAI_KEY \
    --force-font sans-serif \
    --files input.pdf
```

日本語のフォントとして BIZ UD Gothic/Mincho を利用します。以下のファイルが `$HOME/.cache/babeldoc/fonts` に置く必要があります。他のフォントと違い、ダウンロードは手動です。

- BIZUDGothic-Regular.ttf
- BIZUDGothic-Bold.ttf
- BIZUDMincho-Regular.ttf
- BIZUDMincho-Bold.ttf

## マルチコラム対応

BabelDOC や他の翻訳器ではテキストをブロック単位で翻訳します。そのため、2-columns テキストで、一文章が二つのコラムに渡る場合、正しい訳出ができません。このブランチでは英文が2つのブロックにまたがっていると思われる場合、それを一つにして翻訳し、その後訳文を元のブロックに按分する hack を入れています。大抵うまくいきますが、いかない場合もあります。

## プロンプト

日本語常体(だ。である。)向けにプロンプトを修正してあります。

## フォント

### Serif/sans-serif フォント選択の workaround

PyMuPDF は serif/sans-serif の認識がおかしい場合があり、その workaround として `--force-font serif/sans-serif` オプションを追加しています。

### 使用する日本語フォントの変更

BIZ UD フォントを日本語に使用しています。

- BIZUDGothic-Regular.ttf
- BIZUDGothic-Bold.ttf
- BIZUDMincho-Regular.ttf
- BIZUDMincho-Bold.ttf

上述のようにこれらのファイルは手動でダウンロードする必要があります。

日本語でない文字については、以下のフォントを利用します:

- NotoSans-Regular.ttf
- NotoSans-Bold.ttf

これらは自動的にダウンロードされます。

## Font subsetting バグの対応

PyMuPDF の font subsetting にバグがあるようで、元の文章部分のフォント情報が一部欠落します。その問題に対処するために、`fonttoolsl` を利用した subsetting を使っています。
