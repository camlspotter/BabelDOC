import dotenv

dotenv.load_dotenv()

import gradio as gr
import gradio_pdf as PDF

from babeldoc.high_level import async_translate
from babeldoc.translation_config import TranslationConfig, TranslateResult
from babeldoc.document_il.translator.translator import OpenAITranslator
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.translation_config import WatermarkOutputMode
import os

from typing import Literal, get_args

Lang = Literal['英語', '日本語']
langs = get_args(Lang)

def lang_id(lang : Lang) -> str:
    match lang:
        case '英語':
            return 'en'
        case '日本語':
            return 'jp'

Font = Literal['なし', 'serif/明朝', 'sans-serif/ゴシック']
fonts = list(get_args(Font))

ConnectColumns = Literal['しない', 'する']
connect_columns = list(get_args(ConnectColumns))

def connect_columns_value(s : ConnectColumns) -> bool:
    match s:
        case 'しない':
            return False
        case 'する':
            return True

def force_serif_value(f : Font) -> bool | None:
    match f:
        case 'なし':
            return None
        case 'serif/明朝':
            return True
        case 'sans-serif/ゴシック':
            return False

def build_config(
    lang_in : Lang,
    lang_out : Lang,
    input_file : str,
    translation_prompt : str,
    font : Font,
    connect_columns : ConnectColumns,
) -> TranslationConfig:
    translator = OpenAITranslator(
        lang_in = lang_id(lang_in),
        lang_out = lang_id(lang_out),
        model= 'gpt-4.1-nano',
        api_key= os.environ.get('OPENAI_API_KEY'),
        ignore_cache = False, # Always ignore it, since prompts may change.
    )
    return  TranslationConfig(
        translator,
        input_file,
        lang_in= lang_id(lang_in),
        lang_out= lang_id(lang_out),
        doc_layout_model= DocLayoutModel.load_onnx(),
        force_serif= force_serif_value(font),
        watermark_output_mode= WatermarkOutputMode.Watermarked,
        output_dir = None,
        working_dir = None,
        translation_prompt = translation_prompt,
        connect_columns = connect_columns_value(connect_columns),
    #         pages: str | None = None,
    #         progress_monitor: ProgressMonitor | None = None,
    #         skip_clean: bool = False,
    #         dual_translate_first: bool = False,
    #         disable_rich_text_translate: bool = False,
    #         enhance_compatibility: bool = False,
    #         report_interval: float = 0.1,
    #         min_text_length: int = 5,
    #         use_alternating_pages_dual: bool = False,
    #         # Add split-related parameters
    #         split_strategy: BaseSplitStrategy | None = None,
    #         table_model=None,
    #         show_char_box: bool = False,
    #         skip_scanned_detection: bool = False,
    #         ocr_workaround: bool = False,
    )

def spec_str(lang_in : Lang, lang_out : Lang, detail : str) -> str:
    return f'{lang_in}を{detail}{lang_out}に翻訳してください'

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=0):
            file = gr.File(
                file_count='single',
                label='翻訳対象PDF',
                file_types= ['.pdf'],
            )
            with gr.Row():
                spec = gr.Markdown(f'指示: {spec_str(langs[0],langs[1],'')}')
                lang_in = gr.Dropdown(
                    langs,
                    label='翻訳元',
                    value=langs[0],
                )
                lang_out = gr.Dropdown(
                    langs,
                    label='翻訳先',
                    value=langs[1],
                )
                details = gr.Text(label='文体指定', placeholder='常体(だ、である)をつかった,簡潔な,礼儀正しい,等')
            font = gr.Radio(
                fonts,
                label='フォント強制',
                value= fonts[0],
            )
            connect_columns = gr.Radio(
                connect_columns,
                label='コラム結合(実験中)',
                value=connect_columns[0],
                
            )
            but = gr.Button('翻訳', interactive=False)
            outputs = gr.Files(label='翻訳結果', visible=False)

        pdf = PDF.PDF(label='プレビュー', scale=1)
    
    def update_but(path : str | None) -> gr.Button:
        return gr.Button(interactive= path is not None)
        
    file.change(update_but,[file],[but])
    file.change(lambda x:x, [file], [pdf])

    def update_spec(lang_in, lang_out, details):
        return spec_str(lang_in,lang_out,details)

    gr.on(
        [lang_in.change, lang_out.change, details.change],
        update_spec,
        [lang_in, lang_out, details],
        spec
    )

    async def doit(
        file, 
        lang_in,
        lang_out,
        details,
        font,
        connect_columns,
        progress= gr.Progress()
    ):
        config = build_config(
            input_file= file,
            lang_in= lang_in,
            lang_out= lang_out,
            translation_prompt= spec_str(lang_in, lang_out, details),
            font= font,
            connect_columns = connect_columns,
        )
        progress(0.0, desc= 'Translating')
        async for event in async_translate(config):
            print(event)
            def report_progress(event):
                if event.get('stage_total'):
                    progress(
                        event['overall_progress'] / 100.0, 
                        desc= f"{event['stage']} ({event['stage_current']}/{event['stage_total']})"
                    )
                else:
                    progress(event['overall_progress'] / 100.0, desc= event['stage'])
            match event['type']:
                case 'progress_update':
                    report_progress(event)
                case 'progress_start':
                    report_progress(event)
                case 'progress_end':
                    report_progress(event)
                case 'finish':
                    progress(1.0, desc='Finished')
                    result : TranslateResult = event['translate_result']
                    return [
                        str(result.mono_pdf_path), # It is Path, not str
                        gr.Button(interactive=True),
                        gr.Files(value=[str(result.mono_pdf_path), str(result.dual_pdf_path)], visible=True)
                    ]
                case _:
                    print(event)
    
    but.click(
        lambda:[gr.Button(interactive=False),
                gr.Files(visible=False)],
        [], [but, outputs]
    ).then(
        doit, 
        [ 
            file,
            lang_in, 
            lang_out, 
            details,
            font, 
            connect_columns
        ],
        [pdf, but, outputs]
    )

def main():
    # Queue() for parallel processing
    demo.queue(
        # default_concurrency_limit=6
    ).launch(
        # max_file_size= '20mb',
        server_name= '0.0.0.0',
        # server_port= 5005,
        # favicon_path= icon.roischat16,
        # root_path='/rois_llm'
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
