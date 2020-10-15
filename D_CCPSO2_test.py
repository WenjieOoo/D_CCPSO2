import numpy as np
import pandas as pd

from mpi4py import MPI
from cec2013lsgo.cec2013 import Benchmark
from D_CCPSO2 import D_CCPSO2_Master, D_CCPSO2_Slave


def main(fun_id):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    bench = Benchmark()

    i_dimension_size = 1000  # 1000
    i_num_slave = 10  # 10
    ifun = bench.get_function(fun_id)
    # ifun = function1
    i_each_slave_evaluations = 500000  # 500000
    i_population_size = 6
    i_x_lower = bench.get_info(fun_id)['lower']
    i_x_upper = bench.get_info(fun_id)['upper']
    # i_x_lower = -100.0
    # i_x_upper = 100.0

    # Master
    if rank == 0:
        imaster = D_CCPSO2_Master(i_comm=comm, fun=ifun, num_subswarm=i_num_slave, dimension_size=i_dimension_size,
                                  x_lower=i_x_lower, x_upper=i_x_upper)
        result = imaster.evolve()
        print("\nFunction Info: " + str(bench.get_info(fun_id)) + '\n')
        print("gbest = " + str(result[-1]))

    # Slave
    else:
        islave = D_CCPSO2_Slave(i_comm=comm, i_rank=rank, fun=ifun,
                                max_number_of_fitness_evaluations=i_each_slave_evaluations,
                                population_size=i_population_size, dimension_size=i_dimension_size, x_lower=i_x_lower,
                                x_upper=i_x_upper)
        islave.evolve()


if __name__ == '__main__':
    main(fun_id=1)
