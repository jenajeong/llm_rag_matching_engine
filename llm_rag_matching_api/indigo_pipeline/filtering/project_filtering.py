import json
from pathlib import Path
from typing import List, Dict, Any
import sys
import re

sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.config import PROJECT_DATA_FILE, DATA_TRAIN_PROJECT_FILE
from indigo_pipeline.filtering.text_preprocessing import preprocess_text


def load_project_json(input_file: str = None) -> List[Dict]:
    if input_file is None:
        input_file = PROJECT_DATA_FILE
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[경고] 파일이 존재하지 않습니다: {input_file}")
        return []
    
    print(f"[파일 읽기] 프로젝트 JSON 파일 읽기 중: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
        
        print(f"  - 총 {len(project_data):,}개의 프로젝트 데이터 로드 완료")
        return project_data
    except Exception as e:
        print(f"  - 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


def parse_year_from_project(project: Dict) -> Any:
    excel_base_year = project.get('excel_기준년도')
    if excel_base_year:
        try:
            year = int(str(excel_base_year).strip())
            if 1900 <= year <= 2100:
                return year
        except:
            pass
    
    rch_st_dt = project.get('RCH_ST_DT')
    if rch_st_dt:
        date_str = str(rch_st_dt).strip()
        if len(date_str) >= 8 and date_str[:4].isdigit():
            try:
                year = int(date_str[:4])
                if 1900 <= year <= 2100:
                    return year
            except:
                pass
        elif '-' in date_str:
            parts = date_str.split('-')
            if len(parts) > 0 and parts[0].isdigit():
                try:
                    year = int(parts[0])
                    if 1900 <= year <= 2100:
                        return year
                except:
                    pass
    
    return None


def deduplicate_by_title_and_professor(data):
    unique = {}
    dup_count = 0

    for item in data:
        title = item.get("title")
        prof = item.get("professor_info")

        title_key = title.strip() if isinstance(title, str) else str(title)
        prof_key = json.dumps(prof, sort_keys=True) if prof else "None"

        key = (title_key, prof_key)

        if key not in unique:
            unique[key] = item
        else:
            dup_count += 1

    return list(unique.values()), dup_count


def filter_project_data(projects: List[Dict]) -> tuple:
    print(f"\n[필터링] 데이터 필터링 시작...")
    print(f"   - 총 프로젝트 수: {len(projects):,}개")
    
    filtered_projects = []
    filter_stats = {
        'total': len(projects),
        'year_filtered': 0,
        'professor_filtered': 0,
        'text_preprocessing_passed': 0,
        'text_preprocessing_failed': 0,
        'dedup_removed': 0
    }
    
    for idx, project in enumerate(projects, 1):
        if idx % 1000 == 0:
            print(f"   - 처리 중: {idx:,}/{len(projects):,}개")
        
        year = parse_year_from_project(project)
        if year is None or year < 2015:
            filter_stats['year_filtered'] += 1
            continue
        
        if not project.get('professor_info'):
            filter_stats['professor_filtered'] += 1
            continue
        
        objective = project.get('excel_연구목표요약', '')
        content = project.get('excel_연구내용요약', '')
        
        summary_parts = []
        if objective and str(objective).strip():
            summary_parts.append(str(objective).strip())
        if content and str(content).strip():
            summary_parts.append(str(content).strip())
        
        if summary_parts:
            summary = ' '.join(summary_parts)
            preprocessed_text, _ = preprocess_text(summary, min_length=0, max_length=5000)
        else:
            preprocessed_text = None
        
        filter_stats['text_preprocessing_passed'] += 1
        
        filtered_project = {
            'data_type': 'project',
            'no': len(filtered_projects) + 1,
            'text': preprocessed_text,
            'title': project.get('PRJ_NM'),
            'year': year,
            'professor_info': project.get('professor_info'),
            'metadata': {
                'PRJ_RSPR_EMP_ID': project.get('PRJ_RSPR_EMP_ID'),
                'TOT_RND_AMT': project.get('TOT_RND_AMT'),
                'RCH_ST_DT': project.get('RCH_ST_DT'),
                'excel_base_year': project.get('excel_기준년도'),
                'excel_project_name_kr': project.get('excel_과제명(국문)'),
                'excel_expected_effect_summary': project.get('excel_기대효과요약'),
                'excel_연구목표요약': project.get('excel_연구목표요약'),
                'excel_연구내용요약': project.get('excel_연구내용요약')
            }
        }
        
        filtered_projects.append(filtered_project)
    
    before_dedup = len(filtered_projects)
    filtered_projects, dedup_removed = deduplicate_by_title_and_professor(filtered_projects)
    after_dedup = len(filtered_projects)

    filter_stats['dedup_removed'] = dedup_removed
    
    print(f"\n[완료] 데이터 필터링 완료")
    print(f"   - 필터링된 프로젝트: {after_dedup:,}개")
    
    return filtered_projects, filter_stats


def save_filtered_data(filtered_projects: List[Dict]):
    train_output_path = Path(DATA_TRAIN_PROJECT_FILE)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[저장] 필터링 후 데이터 저장 중: {train_output_path}")
    
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_projects, f, ensure_ascii=False, indent=2)
        
        print(f"[완료] 총 {len(filtered_projects):,}개의 필터링된 프로젝트 데이터를 저장했습니다.")
        print(f"[저장 위치] {train_output_path}")
    except Exception as e:
        print(f"[오류] 저장 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n[시작] data 폴더의 프로젝트 데이터 읽기 중...")
    projects = load_project_json()
    
    if not projects:
        print("[경고] 프로젝트 데이터가 없습니다.")
        return
    
    original_count = len(projects)
    
    filtered_projects, filter_stats = filter_project_data(projects)
    
    filtered_count = len(filtered_projects)
    
    save_filtered_data(filtered_projects)
    
    print("\n" + "=" * 60)
    print("[통계] 필터링 통계")
    print("=" * 60)
    print(f"1. 원본 프로젝트 수: {original_count:,}개")
    print(f"2. 필터링된 프로젝트 수: {filtered_count:,}개")
    
    professor_matched = len([p for p in filtered_projects if p.get("professor_info")])
    print(f"3. 교수 정보가 있는 프로젝트 수: {professor_matched:,}개")
    
    print(f"\n[연도 필터 통계]")
    print(f"  - 연도 미충족 제외: {filter_stats['year_filtered']:,}개")
    
    print(f"\n[교수 매핑 통계]")
    print(f"  - 교수 정보 없음 제외: {filter_stats['professor_filtered']:,}개")
    
    print(f"\n[중복 제거 통계]")
    print(f"  - title + professor 동일 제거: {filter_stats['dedup_removed']:,}개")
    
    print(f"\n[텍스트 전처리 통계]")
    print(f"  - 전처리 통과: {filter_stats['text_preprocessing_passed']:,}개")
    print(f"  - 전처리 실패: {filter_stats['text_preprocessing_failed']:,}개")
    
    total_filtered = filter_stats['year_filtered'] + filter_stats['professor_filtered']
    print(f"\n[전체 제외]")
    print(f"  - 전체 제외: {total_filtered:,}개 (연도 + 교수)")
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
