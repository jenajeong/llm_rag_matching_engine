import json
from pathlib import Path
from typing import List, Dict, Any
import sys
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

from indigo_pipeline.config import PATENT_DATA_FILE, DATA_TRAIN_PATENT_FILE
from indigo_pipeline.filtering.text_preprocessing import preprocess_text


def load_patent_json(input_file: str = None) -> List[Dict]:
    if input_file is None:
        input_file = PATENT_DATA_FILE
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[경고] 파일이 존재하지 않습니다: {input_file}")
        return []
    
    print(f"[파일 읽기] 특허 JSON 파일 읽기 중: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            patent_data = json.load(f)
        
        print(f"  - 총 {len(patent_data):,}개의 특허 데이터 로드 완료")
        return patent_data
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


def deduplicate_by_title_and_professor(data):
    unique = {}
    for item in data:
        title = item.get("title")
        prof = item.get("professor_info")
        title_key = title.strip() if isinstance(title, str) else str(title)
        prof_key = json.dumps(prof, sort_keys=True) if prof else "None"
        key = (title_key, prof_key)
        if key not in unique:
            unique[key] = item
    return list(unique.values())


def filter_patent_data(patents: List[Dict]) -> tuple:
    print(f"\n[필터링] 데이터 필터링 시작...")
    print(f"   - 총 특허 수: {len(patents):,}개")
    
    filtered_patents = []
    filter_stats = {
        'total': len(patents),
        'title_filtered': 0,
        'text_preprocessing_passed': 0,
        'text_preprocessing_failed': 0,
        'text_over_5000_filtered': 0
    }

    over_5000_samples = []
    
    for idx, patent in enumerate(patents, 1):
        if idx % 1000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(patents):,}개")
        
        kipris_abstract = patent.get('kipris_abstract')

        if has_value(kipris_abstract):
            preprocessed_text, is_valid = preprocess_text(kipris_abstract, min_length=0, max_length=5000)

            if not is_valid:
                filter_stats['text_preprocessing_failed'] += 1
                filter_stats['text_over_5000_filtered'] += 1

                title_for_log = patent.get('kipris_application_name') or patent.get('tech_nm')
                if len(over_5000_samples) < 10:
                    over_5000_samples.append({
                        "title": title_for_log,
                        "original_length": len(str(kipris_abstract))
                    })

                continue
        else:
            preprocessed_text = None

        if not has_value(patent.get('professor_info')):
            continue
        
        title = patent.get('kipris_application_name') or patent.get('tech_nm')

        #  (title null 제거)
        if title is None or str(title).strip() == "":
            filter_stats['title_filtered'] += 1
            continue
        
        filter_stats['text_preprocessing_passed'] += 1
        
        filtered_patent = {
            'data_type': 'patent',
            'no': len(filtered_patents) + 1,
            'text': preprocessed_text,
            'title': title,
            'year': None,
            'professor_info': patent.get('professor_info'),
            'metadata': {
                'mbr_sn': patent.get('mbr_sn'),
                'tech_invnt_se': patent.get('tech_invnt_se'),
                'kipris_register_status': patent.get('kipris_register_status'),
                'kipris_application_date': patent.get('kipris_application_date'),
                'inu_tech_stage_se': patent.get('inu_tech_stage_se')
            }
        }
        
        filtered_patents.append(filtered_patent)
    
    filtered_patents = deduplicate_by_title_and_professor(filtered_patents)
    
    print(f"\n[완료] 데이터 필터링 완료")
    print(f"   - 필터링된 특허: {len(filtered_patents):,}개")
    print(f"   - 5000자 초과 제거: {filter_stats['text_over_5000_filtered']:,}개")

    if over_5000_samples:
        print("\n[5000자 초과 제거 샘플]")
        for i, sample in enumerate(over_5000_samples, 1):
            print(f"   {i}. 길이 {sample['original_length']:,}자 / {sample['title']}")
    
    return filtered_patents, filter_stats


def save_filtered_data(filtered_patents: List[Dict]):
    train_output_path = Path(DATA_TRAIN_PATENT_FILE)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[저장] 필터링 후 데이터 저장 중: {train_output_path}")
    
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_patents, f, ensure_ascii=False, indent=2)
        
        print(f"[완료] 총 {len(filtered_patents):,}개의 필터링된 특허 데이터를 저장했습니다.")
        print(f"[저장 위치] {train_output_path}")
    except Exception as e:
        print(f"[오류] 저장 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n[시작] data 폴더의 특허 데이터 읽기 중...")
    patents = load_patent_json()
    
    if not patents:
        print("[경고] 특허 데이터가 없습니다.")
        return
    
    original_count = len(patents)
    
    filtered_patents, filter_stats = filter_patent_data(patents)
    
    filtered_count = len(filtered_patents)
    
    save_filtered_data(filtered_patents)
    
    print("\n" + "=" * 60)
    print("[통계] 필터링 통계")
    print("=" * 60)
    print(f"1. 원본 특허 수: {original_count:,}개")
    print(f"2. 필터링된 특허 수: {filtered_count:,}개")
    
    print(f"\n[title 필터 통계]")
    print(f"  - title 없음 제거: {filter_stats['title_filtered']:,}개")

    print(f"\n[text 필터 통계]")
    print(f"  - 5000자 초과 제거: {filter_stats['text_over_5000_filtered']:,}개")
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
