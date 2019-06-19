OBJ=../obj
BIN=../bin

MF90TOOLS = /Users/andrewv/projects/f90tools/mod
LF90TOOLS = -L/Users/andrewv/projects/f90tools/lib -lf90tools 

FLAGS=-O5 -J ../mod -I $(MF90TOOLS) -Wuninitialized -fexternal-blas -funroll-loops -fopenmp
FC=gfortran -c $(FLAGS)
FLINK=gfortran $(FLAGS)
#LBLAS=-L/opt/acml4.4.0/gfortran64/lib -lacml -lacml_mv
LBLAS=-lblas

.PHONY: all mdn_test mdn_train mdn_apply

all: mdn_train mdn_apply

# mdn_test: $(BIN)/mdn_test
mdn_train: $(BIN)/mdn_train
mdn_apply: $(BIN)/mdn_apply


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

$(BIN)/mdn_train: $(OBJ)/ziggurat.o $(OBJ)/kmeans.o $(OBJ)/linpack.o \
										$(OBJ)/timer.o $(OBJ)/lbgfsb.o $(OBJ)/mlp.o $(OBJ)/mdn.o $(OBJ)/mdn_train.o
	$(FLINK) $+ -o $@ $(LBLAS) $(LF90TOOLS)

$(BIN)/mdn_apply: $(OBJ)/ziggurat.o $(OBJ)/kmeans.o $(OBJ)/linpack.o \
										$(OBJ)/timer.o $(OBJ)/lbgfsb.o $(OBJ)/mlp.o $(OBJ)/mdn.o $(OBJ)/mdn_apply.o
	$(FLINK) $+ -o $@ $(LBLAS) $(LF90TOOLS)

# $(BIN)/mdn_test: $(OBJ)/ziggurat.o $(OBJ)/kmeans.o $(OBJ)/linpack.o \
# 										$(OBJ)/timer.o $(OBJ)/lbgfsb.o $(OBJ)/mlp.o $(OBJ)/mdn.o $(OBJ)/mdn_test.o
# 	$(FLINK) $+ -o $@ $(LBLAS)
