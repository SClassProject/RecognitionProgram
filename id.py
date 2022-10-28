# id.py

import sys
# 프로그램 실행시 인자값 받아옴
str = sys.argv[1]

if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()

#id만 잘라냄
id = str.split('/')[2]
room_id = str.split('/')[3]