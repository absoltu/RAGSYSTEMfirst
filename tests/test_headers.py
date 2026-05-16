"""
Test script to verify header extraction and chunk enrichment.
"""

from utils.chunk_sectioning import ChunkSectionMapper

# Test markdown text with headers
test_text = """
# Глава 1: Введение

Это введение к документу.

## 1.1 Основные понятия

Здесь описаны основные понятия.

## 1.2 Определения

Здесь даны определения.

# Глава 2: Теория

Начало теории.

## 2.1 Принципы

Принципы системы.

### 2.1.1 Первый принцип

Описание первого принципа.

### 2.1.2 Второй принцип

Описание второго принципа.

## 2.2 Применение

Применение теории.

# Глава 3: Практика

Практическое применение.
"""

# Create mapper
mapper = ChunkSectionMapper(add_section_prefix=True)

# Extract headers
headers = mapper.extract_headers_hierarchy(test_text)

print("=" * 80)
print("EXTRACTED HEADERS:")
print("=" * 80)

for h in headers:
    print(f"Level {h['level']}: {h['title']}")
    print(f"  Position: {h['start_pos']} - {h['end_pos']}")
    print()

# Create test chunks
chunks = [
    "Это введение к документу.",
    "Здесь описаны основные понятия.",
    "Здесь даны определения.",
    "Начало теории.",
    "Принципы системы.",
    "Описание первого принципа.",
    "Описание второго принципа.",
    "Применение теории.",
    "Практическое применение."
]

print("=" * 80)
print("MAPPING CHUNKS TO HEADERS:")
print("=" * 80)

enhanced_chunks, chunk_metadata = mapper.map_chunks_to_headers(
    chunks, 
    test_text, 
    headers
)

for i, (orig, enhanced, meta) in enumerate(zip(chunks, enhanced_chunks, chunk_metadata)):
    print(f"\nCHUNK {i}:")
    print(f"Original: {orig[:50]}...")
    print(f"Section: {meta['section']}")
    print(f"Headers: {meta['headers']}")
    print(f"Enhanced:\n{enhanced[:100]}...\n")
    print("-" * 40)
