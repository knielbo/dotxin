import pinyin

s = "你 好"

print(pinyin.get(s))
print()
print(pinyin.get(s, format="strip"))
