all: 
	nvcc -o Naive Programs/MatrixMultNaive.cu
	nvcc -o Tiled Programs/MatrixMultTiled.cu
	nvcc -o Atoms Programs/MatrixMultAtoms.cu
	nvcc -o Streams Programs/MatrixMultStreams.cu
	nvcc -o StreamedAtoms Programs/MatrixMultStremedAtoms.cu
run: all
	./Naive 5
	./Tiled 5
	./Atoms 5
	./Streams 5
	./StreamedAtoms 5

	./Naive 10
	./Tiled 10
	./Atoms 10
	./Streams 10
	./StreamedAtoms 10

	./Naive 50
	./Tiled 50
	./Atoms 50
	./Streams 50
	./StreamedAtoms 50

	./Naive 100
	./Tiled 100
	./Atoms 100
	./Streams 100
	./StreamedAtoms 100
clean:
	rm -f Naive 
	rm -f Tiled
	rm -f Atoms
	rm -f Streams
	rm -f StreamedAtoms
	rm -f CSV/Naive.csv
	rm -f CSV/Tiled.csv
	rm -f CSV/Atomics.csv
	rm -f CSV/Streams.csv
	rm -f CSV/StreamedAtomics.csv
	rm -f SpeedUpImages/speedup_comparison.png
	rm -f SpeedUpImages/time_comparison.png