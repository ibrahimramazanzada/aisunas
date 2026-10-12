"""Integration and unit tests for the ingestion pipeline.

Tests PDF extraction, Azerbaijani diacritics normalization, article parsing,
and spot-check validation.
"""

import sys
import tempfile
from pathlib import Path
import pymupdf

from src.ingestion.article_parser import ArticleParser
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.text_cleaner import (
    is_suspicious_text,
    normalize_azerbaijani_text,
)
from src.ingestion.validate_parser import validate_articles_file


def create_sample_azerbaijani_pdf(output_path: Path) -> Path:
    """Generates a sample PDF containing official Azerbaijani Labour Code text.

    Args:
        output_path: Destination path for the generated PDF.

    Returns:
        Path: The created PDF file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    font_path = next((p for p in font_candidates if p.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("A Unicode font is required for this test.")
    doc = pymupdf.open()

    page1 = doc.new_page()
    page1.insert_font(fontname="arial", fontfile=str(font_path))

    content_p1 = (
        "AZƏRBAYCAN RESPUBLİKASININ ƏMƏK MƏCƏLLƏSİ\n\n"
        "BÖLMƏ I. ÜMUMİ QAYDALAR\n\n"
        "FƏSİL 10. ƏMƏK MÜQAVİLƏSİNƏ XİTAM VERİLMƏSİ\n\n"
        "Maddə 70. Əmək müqaviləsinin işəgötürən tərəfindən ləğv "
        "edilməsinin əsasları\n\n"
        "Əmək müqaviləsi işəgötürən tərəfindən aşağıdakı əsaslarla "
        "ləğv edilə bilər:\n"
        "a) müəssisə ləğv edildikdə;\n"
        "b) işçilərin sayı və ya ştatları ixtisarı edildikdə;\n"
        "c) peşəkarlıq səviyyəsinin yetərli olmaması nəticəsində "
        "işçinin tutduğu vəzifəyə uyğun gəlmədiyi aşkar edildikdə;\n"
        "ç) işçi tərəfindən əmək funksiyalarının kobud şəkildə "
        "pozulması hallarına yol verildikdə."
    )
    page1.insert_text(
        (50, 60), content_p1, fontname="arial", fontsize=10
    )

    page2 = doc.new_page()
    page2.insert_font(fontname="arial", fontfile=str(font_path))
    content_p2 = (
        "Maddə 74. Tərəflərin iradəsindən asılı olmayan hallarda əmək "
        "müqaviləsinə xitam verilməsinin əsasları\n\n"
        "1. Əmək müqaviləsinə tərəflərin iradəsindən asılı olmayan aşağıdakı "
        "hallarda xitam verilir:\n"
        "a) işçi həqiqi hərbi xidmətə çağırıldıqda;\n"
        "b) işçinin əvvəlki işinə bərpa edilməsi barədə məhkəmə qərarı "
        "qanuni qüvvəyə mindikdə;\n"
        "c) işçi müəyyən müddətə azadlıqdan məhrum edildikdə;\n"
        "ç) işçi tam əmək qabiliyyətini itirdikdə."
    )
    page2.insert_text(
        (50, 60), content_p2, fontname="arial", fontsize=10
    )

    doc.save(str(output_path))
    doc.close()
    return output_path


def _run_ingestion_test(output_dir: Path):
    """Tests PDF extraction, article parsing, and validation end-to-end."""
    sample_pdf_path = output_dir / "sample_emek_mecellesi.pdf"
    output_json = output_dir / "articles.json"
    output_jsonl = output_dir / "articles.jsonl"
    create_sample_azerbaijani_pdf(sample_pdf_path)

    # 1. Extract
    extractor = PDFExtractor(sample_pdf_path)
    pages = extractor.extract_pages()
    assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"

    # 2. Parse
    parser = ArticleParser()
    articles = parser.parse_pages(pages)
    assert len(articles) == 2, f"Expected 2 articles, got {len(articles)}"

    art70 = next(a for a in articles if a.article_number == "70")
    assert "ləğv edilməsinin əsasları" in art70.title
    assert "ixtisarı" in art70.content
    assert not art70.is_suspicious

    art74 = next(a for a in articles if a.article_number == "74")
    assert "iradəsindən asılı olmayan" in art74.title
    assert "hərbi" in art74.content
    assert not art74.is_suspicious

    # 3. Save
    parser.save_records(
        articles,
        output_json=output_json,
        output_jsonl=output_jsonl,
    )
    assert output_json.is_file()
    assert output_jsonl.is_file()

    # 4. Spot check validation
    passed, report = validate_articles_file(output_json)
    for line in report:
        print(line)
    assert passed, "Validation spot checks failed!"
    print("ALL TESTS PASSED SUCCESSFULLY!")


def test_ingestion_end_to_end():
    """Runs fixture ingestion without overwriting the real project dataset."""
    with tempfile.TemporaryDirectory() as directory:
        _run_ingestion_test(Path(directory))


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    test_ingestion_end_to_end()
