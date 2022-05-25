import math
import numpy as np


def calc_overlap_nmi(num_vertices, result_comm_list, ground_truth_comm_list):
    return OverlapNMI(num_vertices, result_comm_list, ground_truth_comm_list).calculate_overlap_nmi()


class OverlapNMI:

    def __init__(self, num_vertices, result_comm_list, ground_truth_comm_list):
        self.x_comm_list = result_comm_list
        self.y_comm_list = ground_truth_comm_list
        self.num_vertices = num_vertices

    def calculate_overlap_nmi(self):

        # h(num)
        def h(num):
            if num > 0:
                return -1 * num * math.log2(num)
            else:
                return 0

        # H(X_i)
        def H_comm(comm):
            prob1 = float(len(comm)) / self.num_vertices
            prob2 = 1 - prob1
            return h(prob1) + h(prob2)

        # H(X)
        def H_cap(cap):
            res = 0.0
            for comm in cap:
                res += H_comm(comm)
            return res

        # H(X_i, Y_j)
        def H_Xi_joint_Yj(comm_x, comm_y):
            intersect_size = float(len(set(comm_x) & set(comm_y)))
            cap_n = self.num_vertices + 4
            prob11 = (intersect_size + 1) / cap_n
            prob10 = (len(comm_x) - intersect_size + 1) / cap_n
            prob01 = (len(comm_y) - intersect_size + 1) / cap_n
            # prob00 = 1 - prob11 - prob10 - prob01
            prob00 = (self.num_vertices - intersect_size + 1) / cap_n

            if (h(prob11) + h(prob00)) >= (h(prob01) + h(prob10)):
                return h(prob11) + h(prob10) + h(prob01) + h(prob00)
            else:
                return H_comm(comm_x) + H_comm(comm_y)

        # H(X_i|Y_j)
        def H_Xi_given_Yj(comm_x, comm_y):
            return float(H_Xi_joint_Yj(comm_x, comm_y) - H_comm(comm_y))

        # H(X_i|Y)  return min{H(Xi|Yj)} for all j
        def H_Xi_given_Y(comm_x, cap_y):
            tmp_H_Xi_given_Yj = []
            for comm_y in cap_y:
                tmp_H_Xi_given_Yj.append(H_Xi_given_Yj(comm_x, comm_y))
            return float(min(tmp_H_Xi_given_Yj))

        # H(Xi|Y)_norm
        def H_Xi_given_Y_norm(comm_x, cap_y):
            return float(H_Xi_given_Y(comm_x, cap_y) / H_comm(comm_x))

        # # H(X|Y)
        # def H_X_given_Y(cap_x, cap_y):
        #     res = 0.0
        #     for comm_x in cap_x:
        #         res += H_Xi_given_Y(comm_x, cap_y)

        #     return res

        # H(X|Y)_norm
        def H_X_given_Y_norm(cap_x, cap_y):
            res = 0.0
            for comm_x in cap_x:
                res += H_Xi_given_Y_norm(comm_x, cap_y)

            return res / len(cap_x)

        def NMI(cap_x, cap_y):
            if len(cap_x) == 0 or len(cap_y) == 0:
                return 0
            return 1 - 0.5 * (H_X_given_Y_norm(cap_x, cap_y) + H_X_given_Y_norm(cap_y, cap_x))

        return NMI(self.x_comm_list, self.y_comm_list)


if __name__ == '__main__':
    print(calc_overlap_nmi(34, [[9, 10, 14, 15, 16, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 26, 25], [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32, 1, 17]], [[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22], [9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]]))

    '''
        # H(X|Y)
        def get_cap_x_given_cap_y(cap_x, cap_y):

            def get_joint_distribution(comm_x, comm_y):
                prob_matrix = np.ndarray(shape=(2, 2), dtype=float)
                intersect_size = float(len(set(comm_x) & set(comm_y)))
                cap_n = self.num_vertices
                prob_matrix[1][1] = (intersect_size) / cap_n
                prob_matrix[1][0] = (len(comm_x) - intersect_size) / cap_n
                prob_matrix[0][1] = (len(comm_y) - intersect_size) / cap_n
                prob_matrix[0][0] = 1 - prob_matrix[1][1] - prob_matrix[1][0] - prob_matrix[0][1]
                return prob_matrix

            # H(Xi)
            def get_single_distribution(comm):
                prob_arr = [0] * 2
                prob_arr[1] = float(len(comm)) / self.num_vertices
                prob_arr[0] = 1 - prob_arr[1]
                return prob_arr

            # H(Xi, Yj)
            def get_cond_entropy(comm_x, comm_y):
                prob_matrix = get_joint_distribution(comm_x, comm_y)
                entropy_list = list(
                    map(
                        OverlapNMI.entropy,
                        (prob_matrix[0][0], prob_matrix[0][1], prob_matrix[1][0], prob_matrix[1][1])))
                if entropy_list[3] + entropy_list[0] <= entropy_list[1] + entropy_list[2]:
                    return np.inf
                else:
                    prob_arr_y = get_single_distribution(comm_y)
                    return sum(entropy_list) - sum(map(OverlapNMI.entropy, prob_arr_y))

            partial_res_list = []
            for comm_x in cap_x:
                cond_entropy_list = map(lambda comm_y: get_cond_entropy(comm_x, comm_y), cap_y)
                min_cond_entropy = float(min(cond_entropy_list))
                partial_res_list.append(
                    min_cond_entropy / sum(map(OverlapNMI.entropy, get_single_distribution(comm_x))))
            return np.mean(partial_res_list)

        return 1 - 0.5 * get_cap_x_given_cap_y(self.x_comm_list, self.y_comm_list) - 0.5 * get_cap_x_given_cap_y(
            self.y_comm_list, self.x_comm_list)
        '''

