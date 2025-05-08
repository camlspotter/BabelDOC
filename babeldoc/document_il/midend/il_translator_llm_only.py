import json
import logging
import re
from pathlib import Path

import Levenshtein
import tiktoken
from tqdm import tqdm

from babeldoc.document_il import Document
from babeldoc.document_il import Page
from babeldoc.document_il import PdfFont
from babeldoc.document_il import PdfParagraph
from babeldoc.document_il.midend.il_translator import DocumentTranslateTracker
from babeldoc.document_il.midend.il_translator import ILTranslator
from babeldoc.document_il.midend.il_translator import PageTranslateTracker
from babeldoc.document_il.translator.translator import BaseTranslator
from babeldoc.document_il.utils.fontmap import FontMapper
from babeldoc.document_il.utils.priority_thread_pool_executor import (
    PriorityThreadPoolExecutor,
)
from babeldoc.translation_config import TranslationConfig

logger = logging.getLogger(__name__)


class BatchParagraph:
    def __init__(
        self, paragraphs: list[PdfParagraph], page_tracker: PageTranslateTracker
    ):
        self.paragraphs = paragraphs
        self.trackers = [page_tracker.new_paragraph() for _ in paragraphs]


def get_head_sentence(s : str) -> tuple[str,str] | None:
    """Get the first full sentence."""
    # Figure 5.1.5 contains '.' but not the end of a sentence
    if g := re.search(r'\.(\s+|$)', s):
        return (s[:g.end()], s[g.end():])
    else:
        return None

def get_head_sentences(s : str) -> tuple[str,str] | None:
    """Get all the full sentences from the head."""
    if g := re.search(r'^.*\.(\s+|$)', s):
        return (g.group(0), s[len(g.group(0)):])
    else:
        return None

def tweak_inputs(inputs : list[tuple[str, ILTranslator.TranslateInput, PdfParagraph, PageTranslateTracker]]):
    """Group sentences seemingly splitted in inputs"""
    tweaked : list[tuple[list[tuple[int,str]],str | None]] = []
    buf = []
    for id_, input_text in enumerate(inputs):
        input = input_text[0]
        layout_label = input_text[2].layout_label
        match layout_label:
            case 'plain text':
                if buf == []:
                    match get_head_sentences(input):
                        case None:
                            buf = [(id_, input)]
                        case (s1, s2):
                            tweaked.append(([(id_, s1)], layout_label))
                            if s2 != "":
                                buf = [(id_, s2)]
                else:
                    match get_head_sentence(input):
                        case None:
                            buf.append((id_, input))
                        case (s1, s2):
                            buf.append((id_, s1))
                            tweaked.append((buf, layout_label))
                            buf = []
                            match get_head_sentences(s2):
                                case None:
                                    if s2 != "":
                                        buf = [(id_, s2)]
                                case (s21, s22):
                                    tweaked.append(([(id_, s21)], layout_label))
                                    if s22 != "":
                                        buf = [(id_, s22)]
            case _:
                tweaked.append(([(id_, input)], layout_label))
    if buf != []:
        tweaked.append((buf, 'plain text'))
    # Check the plain text contents are preserved.
    org_contents = ''.join([input_text[0] for input_text in inputs if input_text[2].layout_label == 'plain text'])
    tweaked_contents = ''.join([ str for (id_str_list, layout_label) in tweaked if layout_label == 'plain text'
                                     for (_, str) in id_str_list ])
    if org_contents != tweaked_contents:
        logger.error(f'Bug of tweak_inputs, org_contents: {org_contents}; tweaked_contents: {tweaked_contents}')
        logger.error(org_contents)
        logger.error(tweaked_contents)
        assert False
    return tweaked

def no_tweak_inputs(
    inputs : list[tuple[str, ILTranslator.TranslateInput, PdfParagraph, PageTranslateTracker]]
) -> list[tuple[list[tuple[int,str]],str | None]]:
    return [ ([(id_,input_text[0])], input_text[0]) for (id_, input_text) in enumerate(inputs) ]

def untweak_translation(
    tweaked : list[tuple[list[tuple[int,str]],str]], 
    translation_results : dict[int,str]
) -> dict[int,str]:
    """Apportion the translation into the original shape of input blocks"""
    translation_results2 : dict[int,str] = {}
    def add_translation(id, s):
        if id in translation_results2:
            translation_results2[id] += ' ' + s
        else:
            translation_results2[id] = s
    # xxx recover if lengths are different

    for (id_, (xs, _layout_label)) in enumerate(tweaked):
        translation = translation_results.get(id_, '')
        total_len = sum([len(s) for (_, s) in xs])
        logger.info(('Translation', [s for (_, s) in xs], translation))
        translation_len = len(translation)
        if total_len == 0:
            # Weird...
            add_translation(xs[0][0], translation)
        else:
            for (pos, (id, s)) in enumerate(xs):
                if pos == len(xs) - 1:
                    l = len(translation)
                else:
                    l = int(translation_len * len(s) / total_len)
                if l == 0:
                    # Weird...
                    pass
                else:
                    add_translation(id, translation[:l])
                    translation = translation[l:]
    return translation_results2

class ILTranslatorLLMOnly:
    stage_name = "Translate Paragraphs"

    def __init__(
        self,
        translate_engine: BaseTranslator,
        translation_config: TranslationConfig,
        tokenizer=None,
    ):
        self.log_oc = open('llm_input_output.log', 'w', encoding='utf-8')
        self.log_prompt = False

        self.translate_engine = translate_engine
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)
        self.shared_context_cross_split_part = (
            translation_config.shared_context_cross_split_part
        )

        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            self.tokenizer = tokenizer

        self.il_translator = ILTranslator(
            translate_engine=translate_engine,
            translation_config=translation_config,
            tokenizer=self.tokenizer,
        )

        try:
            self.translate_engine.do_llm_translate(None)
        except NotImplementedError as e:
            raise ValueError("LLM translator not supported") from e

    def calc_token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return 0

    def find_title_paragraph(self, docs: Document) -> PdfParagraph | None:
        """Find the first paragraph with layout_label 'title' in the document.

        Args:
            docs: The document to search in

        Returns:
            The first title paragraph found, or None if no title paragraph exists
        """
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.layout_label == "title":
                    logger.info(f"Found title paragraph: {paragraph.unicode}")
                    return paragraph
        return None

    def translate(self, docs: Document) -> None:
        tracker = DocumentTranslateTracker()

        if not self.translation_config.shared_context_cross_split_part.first_paragraph:
            # Try to find the first title paragraph
            title_paragraph = self.find_title_paragraph(docs)
            self.translation_config.shared_context_cross_split_part.first_paragraph = (
                title_paragraph
            )
            self.translation_config.shared_context_cross_split_part.recent_title_paragraph = title_paragraph
            if title_paragraph:
                logger.info(f"Found first title paragraph: {title_paragraph.unicode}")

        # count total paragraph
        total = sum(
            [
                len(
                    [
                        p
                        for p in page.pdf_paragraph
                        if p.debug_id is not None and p.unicode is not None
                    ]
                )
                for page in docs.page
            ]
        )
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            with PriorityThreadPoolExecutor(
                max_workers=self.translation_config.qps,
            ) as executor2:
                with PriorityThreadPoolExecutor(
                    max_workers=self.translation_config.qps,
                ) as executor:
                    for page in docs.page:
                        self.process_page(
                            page,
                            executor,
                            pbar,
                            tracker.new_page(),
                            executor2,
                        )

        path = self.translation_config.get_working_file_path("translate_tracking.json")

        if self.translation_config.debug:
            logger.debug(f"save translate tracking to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                f.write(tracker.to_json())
        
        self.log_oc.close()

    def process_page(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: PageTranslateTracker = None,
        executor2: PriorityThreadPoolExecutor | None = None,
    ):
        self.translation_config.raise_if_cancelled()
        page_font_map = {}
        for font in page.pdf_font:
            page_font_map[font.font_id] = font
        page_xobj_font_map = {}
        for xobj in page.pdf_xobject:
            page_xobj_font_map[xobj.xobj_id] = page_font_map.copy()
            for font in xobj.pdf_font:
                page_xobj_font_map[xobj.xobj_id][font.font_id] = font

        paragraphs = []

        total_token_count = 0
        for paragraph in page.pdf_paragraph:
            if paragraph.debug_id is None or paragraph.unicode is None:
                continue
            # self.translate_paragraph(paragraph, pbar,tracker.new_paragraph(), page_font_map, page_xobj_font_map)
            total_token_count += self.calc_token_count(paragraph.unicode)
            paragraphs.append(paragraph)
            if paragraph.layout_label == "title":
                self.shared_context_cross_split_part.recent_title_paragraph = paragraph

            if total_token_count > 400 or len(paragraphs) > 5:
                executor.submit(
                    self.translate_paragraph,
                    BatchParagraph(paragraphs, tracker),
                    pbar,
                    page_font_map,
                    page_xobj_font_map,
                    self.translation_config.shared_context_cross_split_part.first_paragraph,
                    self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
                    executor2,
                    priority=1048576 - total_token_count,
                    paragraph_token_count=total_token_count,
                )
                paragraphs = []
                total_token_count = 0

        if paragraphs:
            executor.submit(
                self.translate_paragraph,
                BatchParagraph(paragraphs, tracker),
                pbar,
                page_font_map,
                page_xobj_font_map,
                self.translation_config.shared_context_cross_split_part.first_paragraph,
                self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
                executor2,
                priority=1048576 - total_token_count,
                paragraph_token_count=total_token_count,
            )

    def translate_paragraph(
        self,
        batch_paragraph: BatchParagraph,
        pbar: tqdm | None = None,
        page_font_map: dict[str, PdfFont] = None,
        xobj_font_map: dict[int, dict[str, PdfFont]] = None,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
        executor: PriorityThreadPoolExecutor | None = None,
        paragraph_token_count: int = 0,
    ):
        """Translate a paragraph using pre and post processing functions."""
        self.translation_config.raise_if_cancelled()
        should_translate_paragraph = []
        try:
            inputs : list[tuple[str, ILTranslator.TranslateInput, PdfParagraph, PageTranslateTracker]] = []

            for i in range(len(batch_paragraph.paragraphs)):
                paragraph : PdfParagraph = batch_paragraph.paragraphs[i]
                tracker = batch_paragraph.trackers[i]
                text, translate_input = self.il_translator.pre_translate_paragraph(
                    paragraph, tracker, page_font_map, xobj_font_map
                )
                if text is None:
                    pbar.advance(1)
                    continue
                should_translate_paragraph.append(i)
                inputs.append((text, translate_input, paragraph, tracker))
            if not inputs:
                return
            json_format_input = []

            tweaked = []
            if self.translation_config.connect_columns:
                try:
                    tweaked = tweak_inputs(inputs)
                except Exception as e:
                    import traceback
                    traceback.print_exception(e)
                    logger.error(e)
                    raise e
            else:
                tweaked = no_tweak_inputs(inputs)
            
            for id_, input in enumerate(tweaked):
                 input_str = ' '.join([s for (_,s) in input[0]])
                 # xxx layout_label sometimes affects the translation.
                 json_format_input.append(
                     {
                         "id": id_,
                         "input": input_str,
                         # "layout_label": input[1]
                     }
                 )

            json_format_input = json.dumps(
                json_format_input, ensure_ascii=False, indent=2
            )
            llm_input = [
                "You are a professional, authentic machine translation engine."
            ]

            if title_paragraph:
                llm_input.append(
                    f"The first title in the full text: {title_paragraph.unicode}"
                )
            if (
                local_title_paragraph
                and local_title_paragraph.debug_id != title_paragraph.debug_id
            ):
                llm_input.append(
                    f"The most similar title in the full text: {local_title_paragraph.unicode}"
                )

            lang_in = self.translation_config.lang_in
            if self.translation_config.lang_in_nl:
                lang_in = self.translation_config.lang_in_nl
            lang_out = self.translation_config.lang_out
            if self.translation_config.lang_out_nl:
                lang_out = self.translation_config.lang_out_nl
            
            llm_input.append(f'''
`id` と `input` のフィールドを持つレコードのリストを翻訳し、翻訳結果を `id` と `output` のフィールドを持つレコードリストにしてください。

入力データ例です:

<example>
```json
[
    {{
        "id": 1,
        "input": <{lang_in}のテキスト1>,
    }},
    {{
        "id": 2,
        "input": <{lang_in}のテキスト2>,
    }},
    ...
]
```
</example>

出力データ例です:

<example>
```json
[
    {{
        "id": 1,
        "output": <テキスト1の{lang_out}への翻訳>,
    }},
    {{
        "id": 2,
        "output": <テキスト2の{lang_out}への翻訳>,
    }},
    ...
]
```
</example>

- {lang_in}を{lang_out}に翻訳してください。
- すでに{lang_out}の部分は{lang_in}に翻訳しては**いけません**。{lang_out}のままにしてください。
''')

            if not self.log_prompt:
                self.log_prompt = True
                self.log_oc.write('\n\n'.join(llm_input))

            llm_input.append(json_format_input)

            final_input = "\n".join(llm_input).strip()

            llm_output = self.translate_engine.llm_translate(
                final_input,
                rate_limit_params={"paragraph_token_count": paragraph_token_count},
            )
            llm_output = llm_output.strip()

            llm_output = self._clean_json_output(llm_output)

            parsed_output_ = json.loads(llm_output)

            parsed_output : list[dict[str,int|str]]

            # parsed_output is often a dict, but we expect a list of dict
            if isinstance(parsed_output_, dict):
                parsed_output = [parsed_output_]
            else:
                parsed_output = parsed_output_

            # Logging
            log = []
            for id_, input in enumerate(tweaked):
                 input_str = ' '.join([s for (_,s) in input[0]])
                 if d := next((d for d in parsed_output if d.get('id', None) == id_), None):
                    log.append({ "id":id_, "input":input_str, "output":d['output'] })
            self.log_oc.write('\n\n')
            self.log_oc.write(json.dumps(log, indent=2, ensure_ascii=False))
            self.log_oc.write('\n\n')

            try:
                translation_results = {item["id"]: item["output"] for item in parsed_output}
            except Exception as e:
                logger.exception(e)
                logger.error(f'parsed_output: {parsed_output}')
                raise e

            try:
                translation_results = untweak_translation(tweaked, translation_results)
            except Exception as e:
                import traceback
                traceback.print_exception(e)
                logger.error(e)

            if len(translation_results) != len(inputs):
                raise Exception(
                    f"Translation results length mismatch. Expected: {len(inputs)}, Got: {len(translation_results)}"
                )

            for id_, output in translation_results.items():
                should_fallback = True
                try:
                    if not isinstance(output, str):
                        logger.warning(
                            f"Translation result is not a string. Output: {output}"
                        )
                        continue

                    id_ = int(id_)  # Ensure id is an integer
                    if id_ >= len(inputs):
                        logger.warning(f"Invalid id {id_}, skipping")
                        continue

                    # Clean up any excessive punctuation in the translated text
                    translated_text = re.sub(r"[. 。…，]{20,}", ".", output)

                    # Get the original input for this translation
                    translate_input = inputs[id_][1]

                    input_unicode = inputs[id_][2].unicode
                    output_unicode = translated_text

                    input_token_count = self.calc_token_count(input_unicode)
                    output_token_count = self.calc_token_count(output_unicode)

                    # Token size of Japanese is usually much bigger than its English translaiton.
                    # if not (0.3 < output_token_count / input_token_count < 3):
                    #     logger.warning(
                    #         f"Translation result is too long or too short. Input: {input_token_count} for '{input_unicode}', Output: {output_token_count} for '{output_unicode}'"
                    #     )
                    #     continue

                    edit_distance = Levenshtein.distance(input_unicode, output_unicode)
                    if edit_distance < 5 and input_token_count > 20:
                        logger.warning(
                            f"Translation result edit distance is too small. distance: {edit_distance}, input: {input_unicode}, output: {output_unicode}"
                        )
                        continue
                    # Apply the translation to the paragraph
                    self.il_translator.post_translate_paragraph(
                        inputs[id_][2],
                        inputs[id_][3],
                        translate_input,
                        translated_text,
                    )
                    should_fallback = False
                    if pbar:
                        pbar.advance(1)
                except Exception as e:
                    logger.exception(f"Error translating paragraph. Error: {e}.")
                    # Ignore error and continue
                    continue
                finally:
                    if should_fallback:
                        logger.warning(
                            f"Fallback to simple translation. paragraph id: {inputs[id_][2].debug_id}"
                        )
                        paragraph_token_count = self.calc_token_count(
                            inputs[id_][2].unicode
                        )
                        executor.submit(
                            self.il_translator.translate_paragraph,
                            inputs[id_][2],
                            pbar,
                            inputs[id_][3],
                            page_font_map,
                            xobj_font_map,
                            priority=1048576 - paragraph_token_count,
                            paragraph_token_count=paragraph_token_count,
                        )


        except Exception as e:
            logger.exception(e)
            logger.warning(f"Error {e} during translation. try fallback")

            if not should_translate_paragraph:
                should_translate_paragraph = list(
                    range(len(batch_paragraph.paragraphs))
                )
            for i in should_translate_paragraph:
                paragraph = batch_paragraph.paragraphs[i]
                tracker = batch_paragraph.trackers[i]
                if paragraph.debug_id is None:
                    continue
                paragraph_token_count = self.calc_token_count(paragraph.unicode)
                executor.submit(
                    self.il_translator.translate_paragraph,
                    paragraph,
                    pbar,
                    tracker,
                    page_font_map,
                    xobj_font_map,
                    priority=1048576 - paragraph_token_count,
                    paragraph_token_count=paragraph_token_count,
                )

    def _clean_json_output(self, llm_output: str) -> str:
        # Clean up JSON output by removing common wrapper tags
        llm_output = llm_output.strip()
        if llm_output.startswith("<json>"):
            llm_output = llm_output[6:]
        if llm_output.endswith("</json>"):
            llm_output = llm_output[:-7]
        if llm_output.startswith("```json"):
            llm_output = llm_output[7:]
        if llm_output.startswith("```"):
            llm_output = llm_output[3:]
        if llm_output.endswith("```"):
            llm_output = llm_output[:-3]
        return llm_output.strip()
