# encoding: utf-8
'''
description : Use a Bezier curve to represent fingerprints , can effectively compress fingerprint
'''

import sys
from Code.description import point_match
import numpy as np
import queue
import cv2
import os
import copy
from Code.description import fit_Curves


def get_keys(d, value):
    for key in d:
        if value in d[key]:
            return key


def split_bezier(points, t):  ###分割贝塞尔曲线
    p11 = (np.array(points[1]) - np.array(points[0])) * t + np.array(points[0])
    p21 = (np.array(points[2]) - np.array(points[1])) * t + np.array(points[1])
    p31 = (np.array(points[3]) - np.array(points[2])) * t + np.array(points[2])

    p12 = (p21 - p11) * t + p11
    p22 = (p31 - p21) * t + p21
    p13 = (p22 - p12) * t + p12

    return [np.array(points[0]), p11, p12, p13, p13, p22, p31, np.array(points[3])]


def cal_sub_bezier(p1, p2, point_line):
    temp_line = list(reversed(point_line[p1])) + point_line[p2]
    dis = np.inf
    temp_list = temp_line[::3]
    if temp_line[-1] not in temp_list:
        temp_list.append(temp_line[-1])
    for set in fit_Curves.fitCurve(np.array(temp_list), 5):
        temp_dis_1 = np.inf
        temp_dis_2 = np.inf
        for po in fit_Curves.bezier_curve_fit(set):
            po_dis_1 = point_match.cal_dis(po, p1)
            po_dis_2 = point_match.cal_dis(po, p2)
            if po_dis_1 < temp_dis_1:
                temp_dis_1 = po_dis_1
            if po_dis_2 < temp_dis_2:
                temp_dis_2 = po_dis_2
        if temp_dis_1 + temp_dis_2 < dis:
            dis = temp_dis_1 + temp_dis_2
            min_set = set
    t_1 = cal_t(min_set, p1)
    control_points_1 = split_bezier(min_set, t_1)
    t_2 = cal_t(control_points_1[4:], p2)
    return split_bezier(control_points_1[4:], t_2)[:4]


def cal_t(points, breakpoint):
    segment = fit_Curves.bezier_curve_fit(points)
    dis = np.inf
    loc = 0
    for po in segment:
        if point_match.cal_dis(po, breakpoint) < dis:
            dis = point_match.cal_dis(po, breakpoint)
            loc = segment.index(po)
            if dis == 0:
                break
    return loc / len(segment)

def thick(pic):
    """
    指纹脊线变粗
    :param path: 细化的图片路径
    :return: 无
    """
    dst = cv2.GaussianBlur(pic, (3, 3), 0)
    ret, binary = cv2.threshold(pic, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 若要细一点可以改为，和下面变细的方法随便选一种就行
    binary = cv2.erode(binary, kernel, iterations=1)  # 若要粗一点，可以增强iterations
    # binary = cv2.dilate(binary, kernel, iterations=1)#若要细一点可以去掉该句注释
    return binary

class Fingerprint(object):
    global direction8
    direction8 = [(-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]

    def __init__(self, image):  # image为非二值化图
        self.image = image

    def get_binary(self):  # 二值化
        ret, img = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        return img

    def get_shape(self):
        return self.get_binary().shape

    def delete_noise(self, threshold=3):
        img = self.get_binary()
        h, w = img.shape
        template = np.array([[0, 0], [0, 0]])
        for i in np.arange(0, h - 2):
            for j in np.arange(0, w - 2):
                if (img[i:i + 2, j:j + 2] == template).all():
                    img[i + 1, j] = 255
                    img[i, j + 1] = 255
        new_img = img.copy()

        for i in np.arange(1, h - 1):
            for j in np.arange(1, w - 1):  # 扫描整副图像
                if img[i, j] == 0:
                    branches = []  # 该点的八领域分支点
                    for pair in direction8:
                        if img[i + pair[0], j + pair[1]] == 0:
                            branches.append((i + pair[0], j + pair[1]))
                    if len(branches) > 2:  # 该点为分叉点
                        flag = np.zeros(img.shape)  # 标记避免重复扫描 1为已扫描过 0为未扫描
                        flag[i, j] = 1  # 该点标记为已扫描

                        for branch in branches:
                            flag[branch[0], branch[1]] = 1  # 该点的每个八领域分支点标记为已扫描

                        for branch in branches:
                            # 对每个分支进行处理 若该分支对应连通域点数小于threshold 则删除该分支对应连通域
                            visit_queue = queue.LifoQueue()  # 深搜栈
                            visit_queue.put(branch)
                            connected_domain = [branch]  # 待定删除的该分支对应连通域点
                            over_threshold = False  # 判断该分支对应连通域点数是否超过threshold
                            while not visit_queue.empty():
                                if over_threshold:  # 如果超过 则删除
                                    break
                                i_t, j_t = visit_queue.get()
                                for pair in direction8:
                                    ii = i_t + pair[0]
                                    jj = j_t + pair[1]
                                    if img[ii, jj] == 0 and flag[ii, jj] == 0:
                                        if len(connected_domain) > threshold:
                                            over_threshold = True
                                            break
                                        flag[ii, jj] = 1
                                        visit_queue.put((ii, jj))
                                        connected_domain.append((ii, jj))
                            if not over_threshold:  # 该分支对应连通域点数未超过threshold 删除该连通域
                                for pos in connected_domain:
                                    new_img[pos[0], pos[1]] = 255

        #####修补去除噪点可能造成的一个点的残缺，导致线断了
        for i in np.arange(1, h - 1):
            for j in np.arange(1, w - 1):  # 扫描整副图像
                if new_img[i, j] == 0:
                    isBranch = 0
                    for pair in direction8:
                        if new_img[i + pair[0], j + pair[1]] == 0:
                            isBranch += 1
                    if isBranch == 1:  # 是断点
                        break_point = []
                        for i_m in range(i - 2, i + 3):
                            for j_m in range(j - 2, j + 3):
                                if new_img[i_m, j_m] == 0:
                                    isBranch_1 = 0
                                    for pair in direction8:
                                        if new_img[i_m + pair[0], j_m + pair[1]] == 0:
                                            isBranch_1 += 1
                                    if isBranch_1 == 1:
                                        break_point.append((i_m, j_m))
                        if len(break_point) == 2:  # 这种情况可能出现线断了
                            ##判断两个点的位置是不是只相隔一个点
                            flag_1 = abs(break_point[1][0] - break_point[0][0]) == 2 and abs(
                                break_point[1][1] - break_point[0][1]) < 3
                            flag_2 = abs(break_point[1][1] - break_point[0][1]) == 2 and abs(
                                break_point[1][0] - break_point[0][0]) < 3

                            if flag_1 or flag_2:
                                repair_point = [(break_point[1][0] + break_point[0][0]) / 2,
                                                (break_point[1][1] + break_point[0][1]) / 2]
                                ##针对各种位置情况求出要补的点的坐标
                                if abs(break_point[0][0] - repair_point[0]) == 0.5:
                                    repair_point[0] = 2 * repair_point[0] - break_point[0][0]
                                if abs(break_point[0][1] - repair_point[1]) == 0.5:
                                    repair_point[1] = 2 * repair_point[1] - break_point[0][1]
                                repair_point = [int(repair_point[0]), int(repair_point[1])]
                                new_img[repair_point[0], repair_point[1]] = 0
        return new_img

    def show_img(self):
        cv2.imshow('origin_img', self.delete_noise())
        # cv2.waitKey(0)
    def get_thick_origin(self):
        # return thick(self.delete_noise())
        return self.delete_noise()

    def connectedDomain(self):
        img = self.delete_noise()
        h, w = img.shape
        # 各个连通域所包含的像素数目的list
        countTotal = []
        # 连通域编号
        labelNum = 0
        # Last in First Out,后进先出队列，相当于栈
        visitQueue = queue.LifoQueue()
        # 以连通域编号代替原像素值，生成带有连通域编号的新图像
        label = np.zeros(img.shape, dtype=np.int)
        # 扫描整副图像
        for i in np.arange(1, h - 1):
            for j in np.arange(1, w - 1):
                # 扫描到未分配连通域的指纹纹路时
                if img[i, j] == 0 and label[i, j] == 0:
                    # 计算像素数
                    count = 1
                    # 压入栈中
                    visitQueue.put((i, j))
                    # 创建新的连通域编号
                    labelNum += 1
                    # 为其分配连通域编号
                    label[i, j] = labelNum
                    # 对栈进行操作
                    while not visitQueue.empty():
                        i_t, j_t = visitQueue.get()
                        # 每当有一个分叉就会多一组，这里有一种特殊情况，遇到椭圆形的那种还认为是一根线
                        # 一个像素的八连通域中如果存在三个或以上像素点，即认为是分叉点
                        isBranch = 0
                        # 计算分叉点
                        for pair in direction8:
                            if img[i_t + pair[0], j_t + pair[1]] == 0:
                                isBranch += 1
                                # 计算
                        for pair in direction8:
                            if label[i_t + pair[0], j_t + pair[1]] == 0 and img[i_t + pair[0], j_t + pair[1]] == 0:
                                if isBranch < 3:
                                    visitQueue.put((i_t + pair[0], j_t + pair[1]))
                                    label[i_t + pair[0], j_t + pair[1]] = labelNum
                                    count += 1
                    countTotal.append(count)
        new_label = label.copy()
        connectedDomainGroup = {}
        for i in range(1, labelNum + 1):
            connectedDomainGroup[i] = []
        # 使输出的点按顺序
        for i in np.arange(1, h - 1):
            for j in np.arange(1, w - 1):
                if label[i, j] != 0:
                    isBranch = 0
                    for pair in direction8:
                        if label[i + pair[0], j + pair[1]] == label[i, j]:
                            isBranch += 1
                    if isBranch == 1:
                        label_number = label[i, j]
                        connectedDomainGroup[label_number].append([i, j])
                        label[i, j] = 0
                        i_m = i
                        j_m = j
                        branch_flag = True
                        while branch_flag:
                            flag = False
                            for pair in direction8:
                                if label[i_m + pair[0], j_m + pair[1]] == label_number:
                                    i_m, j_m = i_m + pair[0], j_m + pair[1]
                                    connectedDomainGroup[label_number].append([i_m, j_m])
                                    label[i_m, j_m] = 0
                                    flag = True
                                    break
                            if not flag:
                                branch_flag = False
                    elif isBranch == 0:
                        connectedDomainGroup[label[i, j]].append([i, j])
                        label[i, j] = 0

        return connectedDomainGroup, new_label

    def image2point(self):  ####分段拟合
        point_set = {}
        count = 0
        connectedDomainGroup, new_label = self.connectedDomain()
        for key in connectedDomainGroup:
            if len(connectedDomainGroup[key]) > 3:
                for set in fit_Curves.fitCurve(np.array(connectedDomainGroup[key][::3]), 5):
                    point_set[count] = set
                    count += 1
        return point_set

    def repair_finger(self):
        point_set = {}
        dict, point_line, side_match_point, edge_point_line = point_match.final(self.image)  # side是边界，edge是边缘
        connectedDomainGroup, new_label = self.connectedDomain()
        # temp_point_line = copy.copy(point_line)
        temp_connectedDomainGroup = copy.copy(connectedDomainGroup)
        # key_edge = []##储存边界断点的key
        key_del = []
        repeat_key = []
        count = len(connectedDomainGroup) + 1
        count_1 = 0
        for key in dict:
            key_1 = get_keys(connectedDomainGroup, [key[0], key[1]])
            key_del.append(key_1)
            if len(dict[key]) == 1:  ####目前只考虑了1对1的情况

                key_2 = get_keys(connectedDomainGroup, [dict[key][0][0], dict[key][0][1]])
                # del connectedDomainGroup[key_2]
                key_del.append(key_2)
                # if point_line[key][-1] in repeat_key:
                #     print(1)

                temp_connectedDomainGroup[count] = list(reversed(point_line[key])) + point_line[dict[key][0]]
                count += 1
                # repeat_key.append(key)
                # repeat_key.append(dict[key][0])
            else:  # 一对二
                key_2 = get_keys(connectedDomainGroup, [dict[key][0][0], dict[key][0][1]])
                key_3 = get_keys(connectedDomainGroup, [dict[key][1][0], dict[key][1][1]])
                # del connectedDomainGroup[key_2]
                # del connectedDomainGroup[key_3]
                key_del.append(key_2)
                key_del.append(key_3)
                temp_connectedDomainGroup[count] = list(reversed(point_line[key])) + point_line[dict[key][0]]
                count += 1
                temp_connectedDomainGroup[count] = [key] + point_line[dict[key][1]]
                count += 1
        for key in side_match_point:
            key_1 = get_keys(connectedDomainGroup, [key[0], key[1]])
            temp_connectedDomainGroup[key_1].insert(0, [side_match_point[key][0][0], side_match_point[key][0][1]])

        for key in temp_connectedDomainGroup:
            if key not in key_del and len(temp_connectedDomainGroup[key]) > 3:
                temp_list = temp_connectedDomainGroup[key][::3]
                if temp_connectedDomainGroup[key][-1] not in temp_list:
                    temp_list.append(temp_connectedDomainGroup[key][-1])
                for set in fit_Curves.fitCurve(np.array(temp_list), 5):
                    # if count_1 == 134:
                    #     print(temp_list)
                    point_set[count_1] = set
                    count_1 += 1
        return point_set



class Point_Set(object):
    def __init__(self, point_set, shape):
        self.point_set = point_set
        self.shape = shape

    def point2image(self):
        new_img = np.ones(self.shape, dtype=np.uint8) * 255
        for key in self.point_set:
            segment = fit_Curves.bezier_curve_fit(self.point_set[key])
            for point in segment:
                new_img[point[0], point[1]] = 0
        return new_img

        return new_img
    def get_thick_trans(self):
        # return thick(self.point2image())
        return self.point2image()

    def show_img(self):
        cv2.imshow('point2image', self.point2image()
                   )
        # cv2.waitKey(0)


if __name__ == "__main__":
    path = sys.argv[1]
    outdir=sys.argv[2]
    try:
        finger_a = Fingerprint(cv2.imread(path, 0))
        origin_img = finger_a.get_thick_origin()
        point_Set_a = finger_a.image2point()
        image_a = Point_Set(point_Set_a, finger_a.get_shape())
        reappear_img = image_a.get_thick_trans()
        image_name = os.path.splitext(os.path.basename(path))[0]

        cv2.imwrite(outdir + image_name + "_origin.png", origin_img)
        cv2.imwrite(outdir + image_name + "_reappear.png", reappear_img)
    except:
        pass

