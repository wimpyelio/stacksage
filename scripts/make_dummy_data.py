"""Generate dummy SEDE data for testing without real CSV."""
import json, hashlib
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

QA = [
    ("How do I use async/await in Python?", "Use async def and await keyword. Example: async def main(): result = await some_coroutine()", "python,asyncio"),
    ("How to merge two dictionaries in Python 3.9?", "Use the | operator: merged = dict1 | dict2. Or use {**dict1, **dict2} for older versions.", "python"),
    ("How to handle exceptions in FastAPI?", "Use HTTPException: raise HTTPException(status_code=404, detail='Not found'). Add exception handlers with @app.exception_handler.", "python,fastapi"),
    ("How to use pandas groupby?", "df.groupby('column').agg({'value': 'sum'}). Chain with reset_index() to flatten the result.", "python,pandas"),
    ("What is a Python decorator?", "A decorator wraps a function to extend behavior. Use @decorator syntax. Example: @functools.wraps preserves metadata.", "python"),
    ("How to use list comprehensions?", "result = [expr for item in iterable if condition]. Faster than for loops for simple transformations.", "python"),
    ("How to connect SQLAlchemy to PostgreSQL?", "engine = create_engine('postgresql://user:pass@host/db'). Use Session for ORM queries.", "python,sqlalchemy"),
    ("How to write pytest fixtures?", "@pytest.fixture def my_fixture(): return SomeObject(). Pass fixture name as test function argument.", "python,pytest"),
    ("How to use Pydantic for validation?", "class Model(BaseModel): field: str. Call Model(field='value'). Raises ValidationError on failure.", "python,pydantic"),
    ("How to read a CSV with pandas?", "df = pd.read_csv('file.csv'). Use dtype= for types, parse_dates= for date columns.", "python,pandas"),
    ("How to use context managers in Python?", "Use with statement: with open('file') as f: data = f.read(). Implement __enter__/__exit__ for custom ones.", "python"),
    ("How to flatten a nested list?", "[item for sublist in nested for item in sublist]. Or use itertools.chain.from_iterable(nested).", "python"),
    ("How to use dataclasses in Python?", "@dataclass class Point: x: float; y: float. Auto-generates __init__, __repr__, __eq__.", "python"),
    ("How to use type hints in Python?", "def func(x: int) -> str: return str(x). Use Optional[T], List[T], Dict[K,V] from typing.", "python"),
    ("How to profile Python code?", "Use cProfile: python -m cProfile script.py. Or line_profiler with @profile decorator.", "python"),
    ("How to use logging in Python?", "import logging; logger = logging.getLogger(__name__); logger.info('message'). Set level with basicConfig.", "python"),
    ("How to use argparse?", "parser = argparse.ArgumentParser(); parser.add_argument('--name'); args = parser.parse_args().", "python"),
    ("How to sort a list of dicts?", "sorted(lst, key=lambda x: x['field']). Use reverse=True for descending.", "python"),
    ("How to use string formatting in Python?", "f-strings: f'{variable}'. .format(): '{}'.format(val). % operator: '%s' % val.", "python"),
    ("How to use enumerate in Python?", "for i, item in enumerate(lst): print(i, item). Start at 1: enumerate(lst, start=1).", "python"),
    ("How to use zip in Python?", "for a, b in zip(list1, list2): pass. Use zip_longest from itertools to handle unequal lengths.", "python"),
    ("How to use defaultdict?", "from collections import defaultdict; d = defaultdict(list); d['key'].append(val).", "python"),
    ("How to use Counter in Python?", "from collections import Counter; c = Counter(lst); c.most_common(10).", "python"),
    ("How to use pathlib?", "from pathlib import Path; p = Path('dir'); p.mkdir(exist_ok=True); (p/'file.txt').write_text('hello').", "python"),
    ("How to use json module?", "json.dumps(obj) to serialize. json.loads(text) to parse. Use indent= for pretty print.", "python"),
    ("How to use requests library?", "resp = requests.get(url); resp.raise_for_status(); data = resp.json().", "python"),
    ("How to use virtual environments?", "python -m venv .venv; source .venv/bin/activate; pip install packages.", "python"),
    ("How to use unittest in Python?", "class TestCase(unittest.TestCase): def test_something(self): self.assertEqual(a, b).", "python"),
    ("How to use generators in Python?", "def gen(): yield 1; yield 2. Use next() or for loop. Memory efficient for large sequences.", "python"),
    ("How to use lambda functions?", "f = lambda x: x*2. Use with map/filter: list(map(lambda x: x*2, lst)).", "python"),
    ("How to use map and filter?", "doubled = list(map(lambda x: x*2, lst)); evens = list(filter(lambda x: x%2==0, lst)).", "python"),
    ("How to use set operations?", "a | b union, a & b intersection, a - b difference, a ^ b symmetric difference.", "python"),
    ("How to use namedtuple?", "Point = namedtuple('Point', ['x','y']); p = Point(1,2); p.x. Or use dataclasses instead.", "python"),
    ("How to use threading in Python?", "t = threading.Thread(target=func); t.start(); t.join(). Use Lock for shared state.", "python"),
    ("How to use multiprocessing?", "with Pool(4) as p: results = p.map(func, items). Better for CPU-bound tasks.", "python"),
    ("How to use functools?", "functools.partial, functools.lru_cache, functools.reduce. @lru_cache speeds up recursive functions.", "python"),
    ("How to write to a file in Python?", "with open('file.txt', 'w') as f: f.write('text'). Use 'a' to append.", "python"),
    ("How to use regex in Python?", "import re; re.match(), re.search(), re.findall(). Use r'pattern' for raw strings.", "python"),
    ("How to use heapq?", "import heapq; heapq.heappush(h, val); heapq.heappop(h). Use for priority queues.", "python"),
    ("How to use bisect for binary search?", "bisect.bisect_left(sorted_list, val) returns insertion point. O(log n).", "python"),
    ("How to implement a singleton?", "Use a class variable or module-level variable. Or __new__ method. Or just use a module.", "python"),
    ("How to use abstract classes?", "from abc import ABC, abstractmethod. class Base(ABC): @abstractmethod def method(self): pass.", "python"),
    ("How to use property decorator?", "@property def x(self): return self._x. @x.setter def x(self, v): self._x = v.", "python"),
    ("How to do dependency injection in Python?", "Pass dependencies as constructor args. Use protocols for interfaces. Avoid global state.", "python,fastapi"),
    ("How to use itertools?", "itertools.chain, product, combinations, permutations, groupby. All return lazy iterators.", "python"),
    ("How to use struct in Python?", "struct.pack('>I', 1234) to binary. struct.unpack('>I', data) to parse.", "python"),
    ("How to use copy in Python?", "copy.copy() for shallow copy. copy.deepcopy() for deep copy.", "python"),
    ("How to use pickle?", "pickle.dumps(obj) to serialize. pickle.loads(data) to deserialize. Use for Python-only.", "python"),
    ("How to use csv module?", "csv.reader(f) to read. csv.DictWriter(f, fieldnames=[]) to write dicts.", "python"),
    ("How to use datetime in Python?", "from datetime import datetime; now = datetime.now(); formatted = now.strftime('%Y-%m-%d').", "python"),
]

records = []
for i, (title, answer, tags) in enumerate(QA):
    qid = str(1000 + i)
    records.append({
        "doc_id":          hashlib.md5(qid.encode()).hexdigest(),
        "question_id":     qid,
        "question_title":  title,
        "question_body":   f"<p>{title}</p>",
        "question_score":  20 + i * 2,
        "tags":            f"<{'><'.join(tags.split(','))}>",
        "creation_date":   "2023-01-01",
        "accepted_answer": f"<p>{answer}</p>",
        "answer_score":    15 + i,
        "question_url":    f"https://stackoverflow.com/q/{qid}",
    })

out = Path("data/raw/sede_export.csv")
import pandas as pd
df = pd.DataFrame(records)
df.rename(columns={
    "question_body": "Body", "question_title": "Title",
    "question_id": "Id", "question_score": "Score",
    "tags": "Tags", "creation_date": "CreationDate",
    "accepted_answer": "AcceptedAnswerBody", "answer_score": "AnswerScore"
}).to_csv(out, index=False)
print(f"✓ Created {len(records)} dummy Q&A pairs → {out}")
