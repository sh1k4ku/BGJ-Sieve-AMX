app: app/*.cpp ./src/libllib.a
	make -C app -j4
	
./src/libllib.a: ./src/*.cpp
	make -C src -j16

clean:
	make -C src clean
	make -C app clean
