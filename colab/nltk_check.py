"""Ensure the NLTK data TextAttack recipes need, and prove it by USING it.

nltk.data.find() is not a reliable test: wordnet downloads fine and still
reports missing, because it ships zipped and find() on corpora/wordnet does not
resolve it. Resource names also moved in NLTK 3.9 -- averaged_perceptron_tagger
became ..._eng, punkt became punkt_tab -- so probing fixed paths fails on a
perfectly good install. Exercising each resource is the only honest check.
"""
import sys
import nltk

def use_stopwords():
    from nltk.corpus import stopwords
    assert stopwords.words("english")

def use_wordnet():
    from nltk.corpus import wordnet as wn
    assert wn.synsets("dog")

def use_tagger():
    assert nltk.pos_tag(["the", "cat"])

def use_tokenizer():
    assert nltk.word_tokenize("the cat sat")

# probe -> packages that might provide it, old and new names both
CHECKS = [
    ("stopwords", use_stopwords, ["stopwords"]),
    ("wordnet",   use_wordnet,   ["wordnet", "omw-1.4"]),
    ("tagger",    use_tagger,    ["averaged_perceptron_tagger",
                                  "averaged_perceptron_tagger_eng"]),
    ("tokenizer", use_tokenizer, ["punkt", "punkt_tab"]),
]

failed = []
for name, probe, pkgs in CHECKS:
    try:
        probe()
        print(f"  {name:10} ok")
        continue
    except Exception:
        pass
    for pkg in pkgs:                       # self-heal rather than instruct
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    try:
        probe()
        print(f"  {name:10} ok (downloaded)")
    except Exception as e:
        failed.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        print(f"  {name:10} FAILED")

if failed:
    print("\n[FAIL] NLTK data unusable:")
    for f in failed:
        print("   ", f)
    sys.exit(1)
print("  all NLTK resources usable")
