from collections import Counter
from indigo_pipeline.stores.vector_store import ChromaVectorStore

DOC_TYPES = ['patent', 'article', 'project']
ITEM_TYPES = ['chunks', 'entities', 'relations']
BATCH = 2000

store = ChromaVectorStore()
found_dup = False

for doc_type in DOC_TYPES:
    print(f'-- {doc_type} --')
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
            print(f'  [WARN] {name}: ID dup {len(dup_ids)}')
            found_dup = True
        else:
            print(f'  [OK] {name}: no ID dup (total {sum(id_counter.values())})')
        if item_type == 'chunks':
            dup_docs = {d: c for d, c in doc_id_counter.items() if c > 1}
            if dup_docs:
                print(f'  [WARN] {name}: doc_id dup {len(dup_docs)}')
                found_dup = True
            else:
                print(f'  [OK] {name}: no doc_id dup (unique docs {len(doc_id_counter)})')

print('RESULT:', 'DUPLICATE FOUND' if found_dup else 'NO DUPLICATES, OK')
