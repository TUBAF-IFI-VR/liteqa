liteqa_c.so: liteqa_c.cpp Makefile
	gcc -Wc++11-extensions -fPIC -lstdc++ -shared -o liteqa_c.so liteqa_c.cpp

liteqa_c: liteqa_c.cpp Makefile
	g++ -Wc++11-extensions -DLITEQA_C_MAIN -fPIC -lstdc++ -o liteqa_c liteqa_c.cpp

clean:
	rm -f liteqa_c.o
	rm -f libliteqa_c.so
	rm -f liteqa_c
