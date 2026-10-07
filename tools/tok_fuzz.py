import os
import json, random, subprocess, sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(os.environ.get('D1_HF_DIR', 'hf'), 'tokenizer.json'))
random.seed(1)
pieces = ["hello", " world", "  ", "\n", "\r\n", "\t", "'s", "'S", "'ll", "'RE", "don't", "123", "4567890", "3.14", "½", "²", "Ⅻ",
          "日本語", "中文字符", "한국어", "Ελληνικά", "русский", "العربية", "हिन्दी", "ไทย", "🙂", "👨‍👩‍👧", "🇧🇷", "​", " ", "　",
          "<|mask|>", "<|im_start|>", "<think>", "</think>", "python", "Python", "pythonic", "Mathias", "<image>", "<|reserved_7|>",
          "!!!", "...", "?!", " (", ")", "[]{}", "@#$%^&*", "—", "–", "«", "»", "€", "$", "%", " -", "--", "==>", "\\", "/", "|",
          "café", "naïve", "Zürich", "ß", "İ", "ǅ", "ﬁ", "́", "é", "ä", "\x00", "\x7f", " ", "\u0085",
          "    indented", "x\n\n\ny", "  \n  ", " \t \n", "A", "Z", "a1b2c3", "ABC123def", "snake_case", "camelCase", "http://x.y/z?q=1"]
cases = []
for _ in range(3000):
    n = random.randint(1, 12)
    cases.append("".join(random.choice(pieces) for _ in range(n)))
for _ in range(500):
    cases.append("".join(chr(random.choice([random.randint(32, 126), random.randint(0x80, 0x2fff), random.randint(0x3000, 0x9fff), random.randint(0x1f300, 0x1faff), random.randint(9, 13)])) for _ in range(random.randint(1, 40))))
inp = "\n".join(json.dumps(c) for c in cases) + "\n"
out = subprocess.run([os.environ.get('D1_BIN', 'target/release/d1'), 'tokenize', '-m', os.environ.get('D1_MODEL', 'd1-omni-600M-F16.gguf')], input=inp, capture_output=True, text=True).stdout.splitlines()
bad = 0
for c, o in zip(cases, out):
    ref = tk.encode(c, add_special_tokens=False).ids
    if json.loads(o) != ref:
        bad += 1
        if bad <= 5: print('MISMATCH', repr(c), ref, o)
print(f'{len(cases)} strings, {bad} mismatches')
