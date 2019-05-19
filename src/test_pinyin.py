import pinyin
import pinyin.cedict

s = "你 好"

print(pinyin.get(s))
print()
print(pinyin.get(s, format="strip"))

s2 = "你"
print(pinyin.get(s2))
print(pinyin.cedict.translate_word(s2))
