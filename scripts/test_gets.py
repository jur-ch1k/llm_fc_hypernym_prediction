import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


from taxoenrich.core import RuWordNet
wn = RuWordNet('wordnets/RuWordNet')

# Стартовый узел из tracking JSON
node = '124124-N'
s = wn.synsets[node]
print('Узел:', s.synset_name, node)
print('Слова:', list(s.synset_words)[:5])
print()
print('--- get_hyponyms (первые 5) ---')
for h in wn.get_hyponyms(node)[:5]:
    print(h['id'], '-', h['name'])
print()
print('--- get_hypernyms ---')
for h in wn.get_hypernyms(node):
    print(h['id'], '-', h['name'])
    with open(f'h_{h["id"]}.json', 'w', encoding='utf-8') as f:
        json.dump(h, f, indent=4, ensure_ascii=False)