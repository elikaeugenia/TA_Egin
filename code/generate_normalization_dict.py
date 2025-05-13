from datasets import load_dataset
import json
import contextlib
import io
import sys

# Redirect output sementara
f = io.StringIO()
with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    # Import library yang diperlukan
    dataset = load_dataset("theonlydo/indonesia-slang", split="train")

# Buat kamus slang-to-formal
slang_dict = {row["slang"]: row["formal"] for row in dataset}

# Simpan ke file JSON
with open("normalization_dict.json", "w", encoding="utf-8") as f:
    json.dump(slang_dict, f, ensure_ascii=False, indent=4)

print("normalization_dict.json berhasil dibuat.")
