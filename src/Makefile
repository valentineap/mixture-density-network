OBJ=../obj
BIN=../bin

FLAGS=-O5 -J ../mod -Wuninitialized -fexternal-blas -funroll-loops -fopenmp
FC=gfortran -c $(FLAGS)
FLINK=gfortran $(FLAGS)
#LBLAS=-L/opt/acml4.4.0/gfortran64/lib -lacml -lacml_mv
LBLAS=-lblas

.PHONY: all mdn_test

all: mdn_test

mdn_test: $(BIN)/mdn_test


$(OBJ)/%.o: %.f90
	$(FC) $+ -o $@

$(OBJ)/lbgfsb.o: Lbfgsb.3.0/lbfgsb.f
	$(FC) $+ -o $@

$(OBJ)/linpack.o: Lbfgsb.3.0/linpack.f
	$(FC) $+ -o $@

$(OBJ)/timer.o: Lbfgsb.3.0/timer.f
	$(FC) $+ -o $@

$(OBJ)/blas.o: Lbfgsb.3.0/blas.f
	$(FC) $+ -o $@

$(BIN)/mdn_test: $(OBJ)/ziggurat.o $(OBJ)/kmeans.o $(OBJ)/linpack.o \
										$(OBJ)/timer.o $(OBJ)/lbgfsb.o $(OBJ)/mlp.o $(OBJ)/mdn.o $(OBJ)/mdn_test.o
	$(FLINK) $+ -o $@ $(LBLAS)
