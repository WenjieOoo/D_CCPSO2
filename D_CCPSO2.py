import numpy as np
import pandas as pd
from mpi4py import MPI
import random
from scipy.stats import cauchy, norm
from tqdm import tqdm
import pydevd_pycharm
port_mapping = [56843, 56842]

class D_CCPSO2_Master(object):
    def __init__(self, i_comm, fun, num_subswarm, dimension_size, x_lower, x_upper, example_num_max=4):

        # mpi通信子
        self.comm = i_comm

        # 优化的问题
        self.fun = fun

        # 停止的标记
        self.stop_counter = 0

        # 子种群的个数(N)
        self.num_subswarm = num_subswarm

        # 优化问题维度
        self.dimension_size = dimension_size

        # 上下界
        self.x_lower = x_lower
        self.x_upper = x_upper

        # example的最大数量
        self.example_num_max = example_num_max

        # # 全局example
        # self.global_example = np.empty([self.example_num_max, self.dimension_size + 1], dtype=np.float64)
        # 全局example
        self.global_example = (self.x_lower + np.random.rand(self.example_num_max, self.dimension_size + 1) * (
                self.x_upper - self.x_lower)).astype(np.float64)
        # 初始化全局example的适应值
        for i in range(self.example_num_max):
            self.global_example[i, -1] = self.fun(self.global_example[i, :-1])

        # # 当前全局example数量
        # self.has_example = 0

    def evolve(self):
        while 1:
            if self.stop_counter == self.num_subswarm:
                break

            # 接收数据的暂存区buf
            data = np.empty(self.dimension_size + 1, dtype=np.float64)

            status = MPI.Status()
            req = self.comm.Irecv(buf=data)
            req.wait(status=status)

            if status.Get_tag() == 1:
                self.stop_counter += 1

            elif status.Get_tag() == 2:
                # if self.has_example < self.example_num_max:
                #     self.global_example[self.has_example] = data.copy()
                #     self.has_example += 1
                # pydevd_pycharm.settrace('localhost', port=port_mapping[0], stdoutToServer=True,
                #                         stderrToServer=True)
                if data[-1] not in self.global_example[:, -1]:
                    example_temp = np.concatenate([self.global_example, [data]])
                    ind = np.argsort(example_temp[:, -1])
                    example_temp = example_temp[ind]
                    self.global_example = example_temp[:-1, :]

            elif status.Get_tag() == 3:
                req2 = self.comm.Isend(self.global_example, dest=status.Get_source())
                req2.wait()

        pd.DataFrame(data=self.global_example[0]).to_csv('results/D_CCPSO2_gbest_example.csv')
        return self.global_example[0]


class D_CCPSO2_Slave(object):
    def __init__(self, i_comm, i_rank, fun, max_number_of_fitness_evaluations, population_size, dimension_size, x_lower,
                 x_upper, example_num_max=4, max_stuck=2):

        # mpi通信子
        self.comm = i_comm

        # Slave代号
        self.rank = i_rank

        # 优化的问题
        self.fun = fun

        # 进行适应度评估的次数
        self.number_of_fitness_evaluations = 0
        self.max_number_of_fitness_evaluations = max_number_of_fitness_evaluations

        # 每一个子种群的粒子数量
        self.population_size = population_size

        # 优化问题维度
        self.dimension_size = dimension_size

        # 子种群大小可选集
        # self.group_size_set = [2, 5, 50, 100, 200]
        # self.group_size_set = [2, 5, 10]
        self.group_size_set = [2, 5, 10, 50, 100, 250]

        # 子种群维度size
        self.s = random.choice(self.group_size_set)
        # 子种群数量
        self.k = int(self.dimension_size / self.s)

        # 维度索引
        self.dimension_indices = list(range(self.dimension_size))

        # 粒子更新方式的选择
        self.p = 0.5

        # 上下界
        self.x_lower = x_lower
        self.x_upper = x_upper

        # 粒子位置
        self.X = (self.x_lower + np.random.rand(self.population_size, self.dimension_size) * (
                self.x_upper - self.x_lower)).astype(np.float64)

        # 粒子个人最好位置
        self.Y = self.X.copy()
        # 粒子相邻最好位置
        self.Y_local = self.X.copy()

        # example的数量
        self.example_num_max = example_num_max
        # Slave获得的榜样vector
        # self.Y_example = self.X[:self.example_num_max, :].copy()
        self.Y_example = np.empty([self.example_num_max, self.dimension_size + 1], dtype=np.float64)

        # 适应值存储
        self.fX = np.ones([self.population_size, self.k]) * np.inf
        self.fY = self.fX.copy()
        # Slave收到Master传来的目前全局最优值
        # self.fY_example = np.inf
        # self.fY_global = np.inf

        # 第一个子节点的进度条
        if self.rank == 1:
            self.pbar = tqdm(range(self.max_number_of_fitness_evaluations), desc="slave rank" + str(self.rank),
                             position=0)

        # 适应度初始化
        for i in range(self.population_size):
            f_this_line = self.fun(self.X[i])
            for j in range(self.k):
                self.fX[i, j] = f_this_line
                self.fY[i, j] = f_this_line

            self.number_of_fitness_evaluations += 1
            if self.rank == 1:
                self.pbar.update(n=1)

        # Y_local初始化
        for j in range(self.k):
            for i in range(self.population_size):
                local_i = self.local_best(j, i)
                for d in range(j * self.s, (j + 1) * self.s):
                    self.Y_local[i, self.dimension_indices[d]] = self.Y[local_i, self.dimension_indices[d]]

        # 全局最优位置未获得提升的轮数
        self.global_dont_change_i = 0
        # 未变更s的轮数
        self.s_dont_change_i = 0
        # xx轮未有提升，将触发：1、维度组合重组，2、子种群维度size重置，xx轮s未变更也将触发重置
        self.max_stuck = max_stuck

        # 历史全局最优适应值
        self.slave_gbest_history = []

    def b_x(self, j, i, example_i):
        particle_vector = self.Y_example[example_i].copy()
        for d in range(j * self.s, (j + 1) * self.s):
            particle_vector[self.dimension_indices[d]] = self.X[i, self.dimension_indices[d]]
        return particle_vector

    def b_y(self, j, i, example_i):
        particle_vector = self.Y_example[example_i].copy()
        for d in range(j * self.s, (j + 1) * self.s):
            particle_vector[self.dimension_indices[d]] = self.Y[i, self.dimension_indices[d]]
        return particle_vector

    def local_best(self, j, i):
        # v_i = self.fun(self.b(j, i, 'y'))
        v_i = self.fY[i, j]
        v_im1 = float('inf')
        v_ip1 = float('inf')
        if i != 0:
            # v_im1 = self.fun(self.b(j, i - 1, 'y'))
            v_im1 = self.fY[i - 1, j]
        if i == 0:
            v_im1 = self.fY[self.population_size - 1, j]
        if i != self.population_size - 1:
            # v_ip1 = self.fun(self.b(j, i + 1, 'y'))
            v_ip1 = self.fY[i + 1, j]
        if i == self.population_size - 1:
            v_ip1 = self.fY[0, j]

        if v_i < v_ip1 and v_i < v_im1:
            return i
        elif v_ip1 < v_im1:
            if i + 1 >= self.population_size:
                return 0
            return i + 1
        else:
            return i - 1

    def evolve(self):
        while 1:

            for j in range(self.k):
                for i in range(self.population_size):
                    for d in range(j * self.s, (j + 1) * self.s):
                        d_update = self.dimension_indices[d]
                        if random.random() <= self.p:
                            self.X[i, d_update] = self.Y[i, d_update] + cauchy.rvs(loc=0, scale=1, size=1) \
                                                  * abs(self.Y[i, d_update] - self.Y_local[i, d_update])
                        else:
                            self.X[i, d_update] = self.Y_local[i, d_update] + norm.rvs(loc=0, scale=1, size=1) \
                                                  * abs(self.Y[i, d_update] - self.Y_local[i, d_update])
                        self.X[i, d_update] = np.clip(self.X[i, d_update], self.x_lower, self.x_upper)

            # 给Master发一个tag=3的request
            req = self.comm.Isend(np.array([], dtype=np.float64), dest=0, tag=3)
            req.wait()
            # 请求最新的example vector
            req = self.comm.Irecv(buf=self.Y_example, source=0)
            req.wait()

            not_improve = True
            example_i = -1
            # example_i = self.example_num_max
            while not_improve:
                example_i += 1
                # example_i -= 1
                # if example_i >= self.example_num_max:
                #     break
                # if example_i < 0:
                #     break
                if example_i >= 1:
                    break
                for j in range(self.k):
                    for i in range(self.population_size):
                        # 适应度计算 (每个子种群的每个粒子)
                        self.fX[i, j] = self.fun(self.b_x(j, i, example_i=example_i)[:-1])
                        self.fY[i, j] = self.fun(self.b_y(j, i, example_i=example_i)[:-1])
                        self.number_of_fitness_evaluations += 2
                        if self.rank == 1:
                            self.pbar.update(n=2)
                        if self.fX[i, j] < self.fY[i, j]:
                            for d in range(j * self.s, (j + 1) * self.s):
                                self.Y[i, self.dimension_indices[d]] = self.X[i, self.dimension_indices[d]]
                            self.fY[i, j] = self.fX[i, j].copy()
                        # if self.fY[i, j] < self.Y_example[example_i, -1]:
                        if self.fY[i, j] < self.Y_example[0, -1]:
                            for d in range(j * self.s, (j + 1) * self.s):
                                self.Y_example[example_i, self.dimension_indices[d]] = self.Y[
                                    i, self.dimension_indices[d]]
                            self.Y_example[example_i, -1] = self.fY[i, j].copy()
                            not_improve = False

            if not_improve:
                self.global_dont_change_i += 1
            else:
                self.global_dont_change_i = 0

            if self.global_dont_change_i >= self.max_stuck:
                random.shuffle(self.dimension_indices)
                self.fX = np.ones([self.population_size, self.k]) * np.inf
                self.fY = self.fX.copy()

            if (self.global_dont_change_i >= self.max_stuck * 2) or (self.s_dont_change_i >= self.max_stuck * 5):
                self.s = random.choice(self.group_size_set)
                self.k = int(self.dimension_size / self.s)
                self.global_dont_change_i = 0
                self.s_dont_change_i = -1
                self.fX = np.ones([self.population_size, self.k]) * np.inf
                self.fY = self.fX.copy()

            self.s_dont_change_i += 1

            for j in range(self.k):
                for i in range(self.population_size):
                    local_i = self.local_best(j, i)
                    for d in range(j * self.s, (j + 1) * self.s):
                        self.Y_local[i, self.dimension_indices[d]] = self.Y[local_i, self.dimension_indices[d]]

            if not not_improve:
                # 把新gbest发给Master(with tag 2)
                req = self.comm.Isend(self.Y_example[example_i], dest=0, tag=2)
                req.wait()
                self.slave_gbest_history.append([self.number_of_fitness_evaluations, self.Y_example[example_i, -1]])

            if self.number_of_fitness_evaluations + (
                    self.k * self.population_size * self.example_num_max * 2) >= self.max_number_of_fitness_evaluations:
                # 给Master发一个tag=1的演化结束消息
                req = self.comm.Isend(np.array([], dtype=np.float64), dest=0, tag=1)
                req.wait()
                break

        if self.rank == 1:
            self.pbar.close()
            tqdm.write("Slave_1's self.number_of_fitness_evaluations = " + str(self.number_of_fitness_evaluations))
            tqdm.write("Slave rank" + str(self.rank) + " complete.\nWaiting for others...")

        pd.DataFrame(data=self.slave_gbest_history).to_csv('results/D_CCPSOw_Slave' + str(self.rank) + '_gbest_history.csv')