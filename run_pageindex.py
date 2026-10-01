import argparse
import os
import json
import tempfile
from pageindex import *
from pageindex.ocr import (DEFAULT_OCR_MODE, IMAGE_EXTENSIONS, OCR_MODES,
                           has_image_extension, image_to_pdf, ocr_pages,
                           page_needs_transcription)
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader, SUMMARY_CONCURRENCY, SUMMARY_MAX_WORDS
from pageindex.tree_optimize import EXPAND_CONCURRENCY

IMAGE_TYPES = ', '.join(sorted(ext.lstrip('.') for ext in IMAGE_EXTENSIONS))


def ocr_page_list(pdf_file, ocr_mode, ocr_model, flash, always=False):
    """(text, tokens) pages after OCR, or None when the PDF text layer
    serves as is (``always`` returns the pages regardless, for images). In flash mode only pages without a usable text layer
    trigger OCR: flash reads the text layer itself, so figure descriptions
    reach the tree in standard mode only."""
    import PyPDF2
    from pageindex.utils import count_tokens
    try:
        texts = [page.extract_text() or ''
                 for page in PyPDF2.PdfReader(pdf_file).pages]
    except Exception:
        if flash:
            return None  # flash validates the file and reports its own error
        raise
    if ocr_mode == 'off' or (flash and ocr_mode == 'auto' and
                             not any(page_needs_transcription(t) for t in texts)):
        return None
    result = ocr_pages(pdf_file, texts, model=ocr_model, mode=ocr_mode)
    if result.page_texts == texts and not always:
        return None
    return [(text, count_tokens(text, model=ocr_model)) for text in result.page_texts]

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str,
                      help=f'Path to the PDF file, or a PNG/JPEG image ({IMAGE_TYPES}) indexed through OCR')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')
    parser.add_argument('--mode', choices=['flash', 'standard'], default='flash',
                      help='Processing mode (default: flash)')
    parser.add_argument('--flash', action='store_true', default=False,
                      help=argparse.SUPPRESS)
    parser.add_argument('--embedded-toc', action=argparse.BooleanOptionalAction, default=None,
                      help='Use the PDF\'s embedded bookmarks when trustworthy (default: on in flash mode)')
    parser.add_argument('--summary', action=argparse.BooleanOptionalAction, default=None,
                      help='Generate node summaries with an LLM (default: on in flash mode)')
    parser.add_argument('--optimize', nargs='?', const='full', choices=['full', 'merge', 'off'],
                      default=None,
                      help='Refine the tree for search cost (default: full in flash mode). '
                           '`merge` for deterministic merge only; `off` to disable')

    parser.add_argument('--ocr', type=lambda value: value.strip().lower(),
                      choices=list(OCR_MODES), default=None,
                      help="OCR through the index model's vision input: auto transcribes pages "
                           'without a text layer (and, in standard mode, describes figure-heavy '
                           'pages), force transcribes every page, off reads the text layer only. '
                           f"One vision call per OCR'd page (default: {DEFAULT_OCR_MODE})")
    parser.add_argument('--ocr-model', type=str, default=None,
                      help='Vision model used for OCR (default: the index model)')

    parser.add_argument('--index-model', type=str, default=None,
                      help='Model used to index the document (overrides config.yaml)')
    parser.add_argument('--model', type=str, default=None,
                      help='(legacy) Same as --index-model')
    parser.add_argument('--summary-model', type=str, default=None,
                      help='Model for node summaries (falls back to config.yaml summary_model, then --index-model, then --model)')
    parser.add_argument('--summary-max-words', type=int, default=None,
                      help=f'Word cap for each model-written node summary; short leaf nodes keep their own text (flash mode; default {SUMMARY_MAX_WORDS})')
    parser.add_argument('--summary-concurrency', type=int, default=None,
                      help=f'Cap on simultaneous indexing model calls per lane (flash mode; default {SUMMARY_CONCURRENCY}, expand tops out at {EXPAND_CONCURRENCY})')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None,
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default=None,
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default=None,
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default=None,
                      help='Whether to add text to the node')
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()
    if args.flash:
        args.mode = 'flash'

    # Validate that exactly one file type is specified
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    for flag, value in (('--ocr', args.ocr), ('--ocr-model', args.ocr_model)):
        if value is not None and not args.pdf_path:
            raise ValueError(f"{flag} requires --pdf_path")
    if args.ocr is None:
        args.ocr = DEFAULT_OCR_MODE
    for flag, value in (('--optimize', args.optimize),
                        ('--embedded-toc', args.embedded_toc),
                        ('--summary', args.summary),
                        ('--summary-max-words', args.summary_max_words),
                        ('--summary-concurrency', args.summary_concurrency)):
        if value is not None and not (args.pdf_path and args.mode == 'flash'):
            raise ValueError(f"{flag} requires Flash mode with --pdf_path")
    if args.optimize is None:
        args.optimize = 'full' if args.mode == 'flash' else 'off'
    if args.pdf_path and args.mode == 'flash':
        for flag, value in (('--toc-check-pages', args.toc_check_pages),
                            ('--max-pages-per-node', args.max_pages_per_node),
                            ('--max-tokens-per-node', args.max_tokens_per_node),
                            ('--if-add-node-id', args.if_add_node_id),
                            ('--if-add-node-summary', args.if_add_node_summary),
                            ('--if-add-doc-description', args.if_add_doc_description),
                            ('--if-add-node-text', args.if_add_node_text)):
            if value is not None:
                raise ValueError(f"{flag} is not supported in flash mode; use --mode standard")

    if args.pdf_path:
        # Validate the document: a PDF, or an image converted to one
        is_image = has_image_extension(args.pdf_path)
        if not (args.pdf_path.lower().endswith('.pdf') or is_image):
            raise ValueError(f"Document must be a PDF (.pdf) or a PNG or JPEG image ({IMAGE_TYPES})")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"File not found: {args.pdf_path}")
        if is_image and args.ocr == 'off':
            raise ValueError("Image files need OCR; drop --ocr off")

        index_model = ConfigLoader().load({k: v for k, v in {
            'index_model': args.index_model,
            'model': args.model,
        }.items() if v is not None}).model
        pdf_file = args.pdf_path
        # A converted image is only read by OCR: an image always gets a
        # page_list and so the standard pipeline, which reads that list (and
        # never this scratch PDF). The directory goes away even when OCR fails.
        with tempfile.TemporaryDirectory(prefix='pageindex-') as scratch:
            if is_image:
                pdf_file = os.path.join(scratch, 'document.pdf')
                image_to_pdf(args.pdf_path, pdf_file)
            page_list = ocr_page_list(pdf_file, args.ocr, args.ocr_model or index_model,
                                      flash=args.mode == 'flash' and not is_image,
                                      always=is_image)
        if is_image:
            pdf_file = args.pdf_path
        if args.mode == 'flash' and page_list is not None:
            # Flash reads the PDF text layer and cannot see OCR'd text.
            print('Document needed OCR; indexing it in standard mode.')
            args.mode = 'standard'
            args.optimize = 'off'

        if args.mode == 'flash':
            from pageindex.flash import page_index_flash
            from pageindex.flash.api import flash_rejection_reason
            summary_model = ConfigLoader().load({k: v for k, v in {
                'summary_model': args.summary_model,
                'index_model': args.index_model,
                'model': args.model,
            }.items() if v is not None}).summary_model
            will_summarize = args.summary if args.summary is not None else True
            toc_with_page_number = page_index_flash(
                pdf_file,
                optimize=args.optimize if args.optimize != 'off' else False,
                optimize_model=summary_model,
                summary_model=summary_model,
                use_embedded_toc=args.embedded_toc if args.embedded_toc is not None else True,
                summary=will_summarize,
                summary_max_words=args.summary_max_words,
                summary_concurrency=args.summary_concurrency,
            )
            reason = flash_rejection_reason(toc_with_page_number,
                                            standard_hint="--mode standard")
            if reason:
                raise ValueError(reason)
            if 'optimize' in toc_with_page_number:
                o = toc_with_page_number['optimize']
                print(f"Optimize: merges={o['merges']} expands={o['expands']}, "
                      f"worst-case search cost "
                      f"{o['before'].get('worst_case_search_complexity')} -> "
                      f"{o['after'].get('worst_case_search_complexity')} pages")
        else:
            # Process PDF file
            user_opt = {
                'index_model': args.index_model,
                'model': args.model,
                'summary_model': args.summary_model,
                'toc_check_page_num': args.toc_check_pages,
                'max_page_num_each_node': args.max_pages_per_node,
                'max_token_num_each_node': args.max_tokens_per_node,
                'if_add_node_id': args.if_add_node_id,
                'if_add_node_summary': args.if_add_node_summary,
                'if_add_doc_description': args.if_add_doc_description,
                'if_add_node_text': args.if_add_node_text,
            }
            opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})
            toc_with_page_number = page_index_main(args.pdf_path, opt, page_list=page_list)

        print('Parsing done, saving to file...')

        # Save results
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        suffix = '_structure'
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}{suffix}.json'
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

        print(f'Tree structure saved to: {output_file}')
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'index_model': args.index_model,
            'model': args.model,
            'summary_model': args.summary_model,
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load({k: v for k, v in user_opt.items() if v is not None})
        
        # if_add_* pass through as given (absent = off, as before this CLI
        # used config.yaml): the PDF defaults there must not switch on LLM
        # passes the markdown CLI never ran.
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=args.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            summary_model=opt.summary_model,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text,
            if_add_node_id=args.if_add_node_id
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
