import json
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.config import ARTICLE_DATA_FILE, DATA_TRAIN_ARTICLE_FILE
from indigo_pipeline.filtering.text_preprocessing import preprocess_text
from langdetect import detect, LangDetectException, DetectorFactory

DetectorFactory.seed = 0


def load_article_json(input_file: str = None) -> List[Dict]:
    if input_file is None:
        input_file = ARTICLE_DATA_FILE
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[경고] 파일이 존재하지 않습니다: {input_file}")
        return []
    
    print(f"[파일 읽기] 논문 JSON 파일 읽기 중: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        
        print(f"  - 총 {len(article_data):,}개의 논문 데이터 로드 완료")
        return article_data
    except Exception as e:
        print(f"  - 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True


def detect_language(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    
    try:
        if len(text.strip()) < 10:
            return None
        
        language = detect(text)
        return language
    except:
        return None


def parse_list_string(text: str) -> Optional[List]:
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed_list = ast.literal_eval(text)
            if isinstance(parsed_list, list):
                return parsed_list
        except:
            pass
    
    return None


def select_abstract_from_list(abstract_list: List) -> str:
    if len(abstract_list) == 0:
        return ""
    
    korean_abstract = None
    english_abstract = None
    other_abstracts = []
    
    for item in abstract_list:
        if item is None:
            continue
        
        item_str = str(item).strip()
        if not item_str:
            continue
        
        lang = detect_language(item_str)
        
        if lang == 'ko':
            korean_abstract = item_str
        elif lang == 'en':
            english_abstract = item_str
        else:
            other_abstracts.append(item_str)
    
    if korean_abstract:
        return korean_abstract
    elif english_abstract:
        return english_abstract
    elif other_abstracts:
        return other_abstracts[0]
    else:
        first_item = abstract_list[0]
        return str(first_item).strip() if first_item else ""


def process_abstract(article: Dict) -> Any:
    abstract = article.get("abstract")
    abstract_description = article.get("abstract_description")
    abstract_translated = article.get("abstract_translated")
    
    selected_abstract = None
    
    if has_value(abstract):
        selected_abstract = abstract
    elif has_value(abstract_description):
        selected_abstract = abstract_description
    elif has_value(abstract_translated):
        selected_abstract = abstract_translated
    
    if selected_abstract is None:
        return None
    
    if isinstance(selected_abstract, list):
        return select_abstract_from_list(selected_abstract)
    
    if isinstance(selected_abstract, str):
        parsed_list = parse_list_string(selected_abstract)
        if parsed_list:
            return select_abstract_from_list(parsed_list)
        else:
            return selected_abstract
    
    return str(selected_abstract) if selected_abstract else None


def parse_year(year_value: Any) -> Optional[int]:
    if year_value is None:
        return None
    
    try:
        if isinstance(year_value, int):
            return year_value if 1900 <= year_value <= 2100 else None
        elif isinstance(year_value, str):
            year_str = year_value.strip()
            if year_str.isdigit():
                year_int = int(year_str)
                return year_int if 1900 <= year_int <= 2100 else None
    except:
        pass
    
    return None


def has_invalid_metadata(metadata: Dict) -> bool:
    if not metadata:
        return True
    
    for value in metadata.values():
        if value is None or (isinstance(value, str) and not value.strip()):
            return True
        
        if isinstance(value, str) and value.strip() == "기타학술지(비정기발행학술지)":
            return True
    
    return False


def filter_article_data(articles: List[Dict]) -> tuple:
    filtered_articles = []
    filter_stats = {
        'total': len(articles),
        'year_filtered': 0,
        'metadata_filtered': 0,
        'abstract_processed': 0,
        'text_preprocessing_passed': 0,
        'text_preprocessing_failed': 0,
        'abstract_from_original': 0,
        'abstract_from_description': 0,
        'abstract_from_translated': 0,
        'abstract_null': 0,
        'text_null_saved': 0
    }

    log_counts = {
        'year_fail': 0,
        'metadata_fail': 0,
        'too_long': 0,
        'null_text': 0,
        'success': 0
    }
    
    for article in articles:
        year = parse_year(article.get('YY'))
        
        if year is None or year < 2015:
            filter_stats['year_filtered'] += 1
            log_counts['year_fail'] += 1
            continue
        
        metadata = {
            'THSS_PATICP_GBN': article.get('THSS_PATICP_GBN'),
            'JRNL_GBN': article.get('JRNL_GBN')
        }
        
        if has_invalid_metadata(metadata):
            filter_stats['metadata_filtered'] += 1
            log_counts['metadata_fail'] += 1
            continue
        
        original_abstract = article.get("abstract")
        original_abstract_desc = article.get("abstract_description")
        original_abstract_trans = article.get("abstract_translated")
        
        processed_abstract = process_abstract(article)
        filter_stats['abstract_processed'] += 1
        
        if has_value(original_abstract):
            filter_stats['abstract_from_original'] += 1
        elif has_value(original_abstract_desc):
            filter_stats['abstract_from_description'] += 1
        elif has_value(original_abstract_trans):
            filter_stats['abstract_from_translated'] += 1
        else:
            filter_stats['abstract_null'] += 1
        
        if processed_abstract is not None:
            preprocessed_text = str(processed_abstract)
        else:
            preprocessed_text = None

        if preprocessed_text is not None and len(preprocessed_text) > 5000:
            filter_stats['text_preprocessing_failed'] += 1
            log_counts['too_long'] += 1
            continue
        
        if preprocessed_text is not None:
            filter_stats['text_preprocessing_passed'] += 1
            log_counts['success'] += 1
        else:
            filter_stats['text_null_saved'] += 1
            log_counts['null_text'] += 1
        
        filtered_article = {
            'data_type': 'article',
            'no': len(filtered_articles) + 1,
            'text': preprocessed_text,
            'title': article.get('THSS_NM'),
            'year': year,
            'professor_info': article.get('professor_info'),
            'metadata': metadata
        }
        
        filtered_articles.append(filtered_article)

    print("\n===== FILTER LOG =====")
    print("year_filtered:", log_counts['year_fail'])
    print("metadata_filtered:", log_counts['metadata_fail'])
    print("text_too_long:", log_counts['too_long'])
    print("text_null:", log_counts['null_text'])
    print("success:", log_counts['success'])
    print("======================\n")
    
    return filtered_articles, filter_stats


def save_filtered_data(filtered_articles: List[Dict]):
    train_output_path = Path(DATA_TRAIN_ARTICLE_FILE)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_articles, f, ensure_ascii=False, indent=2)


def main():
    articles = load_article_json()
    
    if not articles:
        return
    
    filtered_articles, filter_stats = filter_article_data(articles)
    save_filtered_data(filtered_articles)


if __name__ == "__main__":
    main()
