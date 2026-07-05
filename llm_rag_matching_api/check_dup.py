from collections import Counter
from indigo_pipeline.stores.vector_store import ChromaVectorStore
 
DOC_TYPES = ['patent', 'article', 'project']
ITEM_TYPES = ['chunks', 'entities', 'relations']
BATCH = 2000
 
store = ChromaVectorStore()
found_dup = False
 
for doc_type in DOC_TYPES:
    print(f'── {doc_type} ──')
    for item_type in ITEM_TYPES:
        name = f'{doc_type}_{item_type}'
        collection = store.collections.get(name)
        if collection is None:
            continue
        total = collection.count()
        if total == 0:
            continue
        id_counter = Counter()
        doc_id_counter = Counter()
        offset = 0
        while offset < total:
            result = collection.get(include=['metadatas'], limit=BATCH, offset=offset)
            id_counter.update(result.get('ids', []))
            if item_type == 'chunks':
                doc_id_counter.update(m.get('doc_id') for m in result.get('metadatas', []) if m)
            offset += BATCH
        dup_ids = {i: c for i, c in id_counter.items() if c > 1}
        if dup_ids:
            print(f'  [경고] {name}: ID 중복 {len(dup_ids)}건')
            found_dup = True
        else:
            print(f'  [정상] {name}: ID 중복 없음 (전체 {sum(id_counter.values())}건)')
        if item_type == 'chunks':
            dup_docs = {d: c for d, c in doc_id_counter.items() if c > 1}
            if dup_docs:
                print(f'  [경고] {name}: doc_id 중복 {len(dup_docs)}건')
                found_dup = True
            else:
                print(f'  [정상] {name}: doc_id 중복 없음 (고유 문서 {len(doc_id_counter)}건)')
 
print('결과:', '중복 발견됨' if found_dup else '중복 없음, 정상')
