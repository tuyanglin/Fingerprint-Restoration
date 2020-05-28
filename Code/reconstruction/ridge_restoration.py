# encoding: utf-8
import numpy as np
import copy
import cv2
import math


def symmetric_point(point, e1, e2):
    v_a = (e2[0] - e1[0], e2[1] - e1[1])
    v_b = (point[0] - e1[0], point[1] - e1[1])
    x = ((v_a[0] ** 2 - v_a[1] ** 2) * v_b[0] + 2 * v_a[0] * v_a[1] * v_b[1]) / \
        (v_a[0] ** 2 + v_a[1] ** 2)
    y = (2 * v_a[0] * v_a[1] * v_b[0] + (v_a[1] ** 2 - v_a[0] ** 2) * v_b[1]) / \
        (v_a[0] ** 2 + v_a[1] ** 2)
    return x + e1[0], y + e1[1]


def fix_control_points(curve1, curve2, c1, c2):

    # if symbol=True, it needs to fix control_points
    symbol = False

    len1, threshold1 = (10, 5) if len(curve1) >= 10 else (-1, len(curve1) // 2)
    len2, threshold2 = (10, 5) if len(curve2) >= 10 else (-1, len(curve2) // 2)

    e1 = curve1[0]
    e2 = curve2[0]

    v_a = (e1[0] - e2[0], e1[1] - e2[1])
    positive_1 = 0
    negative_1 = 0
    for p in curve1[1:len1]:
        v_b = (p[0] - e1[0], p[1] - e1[1])
        if v_a[0] * v_b[1] - v_a[1] * v_b[0] > 0:
            positive_1 += 1
        else:
            negative_1 += 1

    v_c = (-v_a[0], -v_a[1])
    positive_2 = 0
    negative_2 = 0
    for p in curve2[1:len2]:
        v_d = (p[0] - e2[0], p[1] - e2[1])
        if v_c[0] * v_d[1] - v_c[1] * v_d[0] > 0:
            positive_2 += 1
        else:
            negative_2 += 1

    direction1 = 1 if positive_1 > negative_1 else -1
    direction2 = 1 if positive_2 > negative_2 else -1

    new_c1 = copy.copy(c1)
    new_c2 = copy.copy(c2)
    # 分别判断两个差距大于阈值
    if math.fabs(positive_1 - negative_1) >= threshold1 and \
            math.fabs(positive_2 - negative_2) >= threshold2 and \
            direction1 * direction2 < 0:
        # 判断左边的控制点需不需要变换
        v_e1_c1 = (c1[0] - e1[0], c1[1] - e1[1])
        if (v_a[0] * v_e1_c1[1] - v_a[1] * v_e1_c1[0]) * direction1 > 0:
            new_c1 = symmetric_point(c1, e1, e2)
            symbol = True
        else:
            # 判断右边的控制点是否需要变换
            v_e2_c2 = (c2[0] - e2[0], c2[1] - e2[1])
            if (v_c[0] * v_e2_c2[1] - v_c[1] * v_e2_c2[0]) * direction2 > 0:
                symbol = True
                new_c2 = symmetric_point(c2, e2, e1)

    elif math.fabs(positive_1 - negative_1) >= threshold1 and \
            math.fabs(positive_2 - negative_2) < threshold2:
        # 判断左边的控制点需不需要变换
        v_e1_c1 = (c1[0] - e1[0], c1[1] - e1[1])
        if (v_a[0] * v_e1_c1[1] - v_a[1] * v_e1_c1[0]) * direction1 > 0:
            new_c1 = symmetric_point(c1, e1, e2)
            symbol = True

    elif math.fabs(positive_1 - negative_1) < threshold1 and \
            math.fabs(positive_2 - negative_2) >= threshold2:
        # 判断右边的控制点是否需要变换
        v_e2_c2 = (c2[0] - e2[0], c2[1] - e2[1])
        if (v_c[0] * v_e2_c2[1] - v_c[1] * v_e2_c2[0]) * direction2 > 0:
            new_c2 = symmetric_point(c2, e2, e1)
            symbol = True

    return new_c1, new_c2, symbol



def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def judge_intersection(e1, c1, e2, c2):
    # if not intersect, return True
    intersection = None
    try:
        intersection = line_intersection((e1, e2),
                                         (c1, c2))
    except Exception as e:
        return True

    if intersection:
        if intersection[0] >= min([e1[0], e2[0]]) and \
                intersection[0] <= max([e1[0], e2[0]]) and \
                intersection[1] >= min([e1[1], e2[1]]) and \
                intersection[1] <= max([e1[1], e2[1]]) and \
                intersection[0] >= min([c1[0], c2[0]]) and \
                intersection[0] <= max([c1[0], c2[0]]) and \
                intersection[1] >= min([c1[1], c2[1]]) and \
                intersection[1] <= max([c1[1], c2[1]]):
            return False

    return True

def dijkstra(pos_list):
    """
    对贝塞尔曲线拟合出来的线进行细化
    :param pos_list:
    :return: 细化后的线
    """
    direction8 = [(-1, 1), (-1, 0), (-1, -1), (0, -1),
                  (1, -1), (1, 0), (1, 1), (0, 1)]
    graph = np.ones((len(pos_list), len(pos_list))) * np.inf

    for pos in pos_list:
        # graph[pos_list.index(pos), pos_list.index(pos)] = 0
        for direction in direction8:
            p = (pos[0] + direction[0], pos[1] + direction[1])
            if p in pos_list:
                graph[pos_list.index(p), pos_list.index(pos)] = \
                    graph[pos_list.index(pos), pos_list.index(p)] = 1

    solved_points = [0]
    unsolved_points = list(range(1, len(pos_list)))

    len_unsolved_points = copy.copy(graph[0])
    num_solved_points = 1
    path = {}
    for p in range(1, len(pos_list)):
        if len_unsolved_points[p] == 1:
            path[p] = [p]
        else:
            path[p] = []

    while num_solved_points < len(pos_list):
        num_solved_points += 1
        shortest_point_id = 1
        len_min = np.inf
        for p in unsolved_points:
            if len_min > len_unsolved_points[p]:
                shortest_point_id = p

        solved_points.append(shortest_point_id)
        # if shortest_point_id in unsolved_points:
        unsolved_points.remove(shortest_point_id)
        for p in range(1, len(pos_list)):
            if len_unsolved_points[p] > \
                    len_unsolved_points[shortest_point_id] + graph[shortest_point_id, p]:
                len_unsolved_points[p] = len_unsolved_points[shortest_point_id] + graph[shortest_point_id, p]
                path[p] = path[shortest_point_id] + [p]

    new_pos_list = [pos_list[0]] + [pos_list[p] for p in path[len(pos_list) - 1]]
    return new_pos_list


def bezier_curve_fit(p1, p2, p3, p4, flag, t_num=2000):
    """
    贝塞尔曲线拟合，需要传入四个点
    :param p1: 端点1
    :param p2: 控制点1
    :param p3: 控制点2
    :param p4: 端点2
    :param t_num: 绘制的点的个数，默认为100个，满足大部分图像大小
    :return: 拟合后的曲线
    """
    bx = 3 * (p2[0] - p1[0])
    cx = 3 * (p3[0] - p2[0]) - bx
    dx = p4[0] - p1[0] - bx - cx
    by = 3 * (p2[1] - p1[1])
    cy = 3 * (p3[1] - p2[1]) - by
    dy = p4[1] - p1[1] - by - cy

    t = np.linspace(0, 1, t_num)

    xx = p1[0] + bx * t + cx * t ** 2 + dx * t ** 3
    yy = p1[1] + by * t + cy * t ** 2 + dy * t ** 3

    segment = list(zip(xx, yy))
    if flag:
        return segment

    segment = [(int(x), int(y)) for (x, y) in segment]
    segment = sorted(set(segment), key=segment.index)

    return dijkstra(segment)


def restoration_l2p(curve, point, control, flag):
    """
    给定一个端点， 一个控制点，一条曲线，进行拟合
    :param curve: 一条曲线
    :param point: 一个端点
    :param control: 一个控制点
    :return: 残缺曲线
    """
    p1 = point
    p2 = control

    t = 2
    if len(curve) < 3:
        t = -1

    p3 = (2 * curve[0][0] - 1 * curve[t][0],
          2 * curve[0][1] - 1 * curve[t][1])
    # p3 = (3 * curve[0][0] - 2 * curve[2][0],
    #       3 * curve[0][1] - 2 * curve[2][1])
    # p3 = (6 * curve[0][0] - 5 * curve[3][0],
    #       6 * curve[0][1] - 5 * curve[3][1])
    p4 = curve[0]

    segment = bezier_curve_fit(p1, p2, p3, p4, flag)
    return segment


def one2one_1p(curve1, curve2, point, flag):
    segment1 = restoration_l2p(curve1, point, point, flag)
    segment2 = restoration_l2p(curve2, point, point, flag)
    segment = segment1[::-1] + segment2
    segment = sorted(set(segment), key=segment.index)
    return dijkstra(segment)


def one2one_2p(curve1, curve2, point1, point2, flag):
    segment1 = restoration_l2p(curve1, point1, point1, flag)
    segment2 = restoration_l2p(curve2, point2, point2, flag)
    segment3 = one2one(segment1 + curve1, segment2 + curve2, flag)
    segment = segment1[::-1] + segment3 + segment2
    segment = sorted(set(segment), key=segment.index)
    return dijkstra(segment)


def get_points(pos_list):
    """
    给定一条曲线，左右两边的控制点， 用于一对二的修复
    :param pos_list:
    :return: 两个控制点
    """
    p1 = None
    if len(pos_list) <= 5:
        p1 = pos_list[-1]
    else:
        p1 = pos_list[5]
    p0 = pos_list[0]
    pa = (p0[0] + p0[1] - p1[1], p1[0] - p0[0] + p0[1])
    pb = (p0[0] - p0[1] + p1[1], p0[0] + p0[1] - p1[0])

    return pa, pb


def get_rate(curve1, curve2):
    i1 = 2 if len(curve1) > 2 else -1
    i2 = 2 if len(curve2) > 2 else -1

    a1 = (curve1[i1][0] - curve1[0][0], curve1[i1][1] - curve1[0][1])
    c1 = (curve2[0][0] - curve1[0][0], curve2[0][1] - curve1[0][1])
    a2 = (curve2[i2][0] - curve2[0][0], curve2[i2][1] - curve2[0][1])
    c2 = (-1 * c1[0], -2 * c1[1])

    theta1 = math.acos(np.dot(a1, c1) / (np.linalg.norm(a1) * (np.linalg.norm(c1))))
    theta2 = math.acos(np.dot(a2, c2) / (np.linalg.norm(a2) * (np.linalg.norm(c2))))

    rate = ((theta1 + theta2) - math.pi) / math.pi

    return rate ** 3


def search(background, pos):
    """
    for partition
    :param background:
    :param pos:
    :return:
    """
    region = background[int(pos[0]) - 1:int(pos[0]) + 2,
             int(pos[1]) - 1:int(pos[1]) + 2]
    if 1 in region:
        return True
    return False




def control_points(curve_one, curve_many_1, curve_many_2):
    control1, control2 = get_points(curve_one)
    flag2 = judge_intersection(curve_many_1[0], curve_many_2[0], control1, control2)
    (c1, c2) = (control1, control2) if flag2 else (control2, control1)

    rate1 = get_rate(curve_many_1, curve_one)

    c1 = (c1[0] - rate1 * c1[0] + rate1 * curve_one[0][0],
          c1[1] - rate1 * c1[1] + rate1 * curve_one[0][1])

    rate2 = get_rate(curve_many_2, curve_one)
    c2 = (c2[0] - rate2 * c2[0] + rate2 * curve_one[0][0],
          c2[1] - rate2 * c2[1] + rate2 * curve_one[0][1])

    return c1, c2


def one2one(curve1, curve2, flag):
    """
    曲线一对一进行修复
    :param curve1:
    :param curve2:
    :return: 残缺曲线段
    """
    p1 = curve1[0]
    p4 = curve2[0]

    p2 = raw_control_point(curve1)
    p3 = raw_control_point(curve2)



    segment = bezier_curve_fit(p1, p2, p3, p4, flag)

    return segment


def raw_control_point(curve):
    """
    initialize control points (one2one)
    :param curve:
    :return:
    """
    if len(curve) < 5:
        points = np.array(curve)
    else:
        points = np.array(curve[:5])
    ctrl_point = 2 * points[0] - points[-1]

    if np.unique(points[:, 0]).size > 1:
        coeff = np.polyfit(points[:, 0], points[:, 1], deg=1)
        ctrl_point[1] = coeff[0] * ctrl_point[0] + coeff[1]

    return tuple(ctrl_point)


def one2one_adjust(curve1, curve2, range, flag, bend=None, ratio=4):
    """
    一对一修复&调整修复曲线弯曲程度
    :param curve1:
    :param curve2:
    :param range: 控制点计算所用比率取值区间
    :param flag:
    :param bend: True-->弯曲程度增加, False-->弯曲程度减小, 默认None-->直接用给定比率计算控制点(Default:4)
    :param ratio: 控制点计算比率，数值越大，控制点越远，曲线曲率越大
    :return:
    """
    step = 3
    if bend == True:
        range[0] = ratio
        while step > 0 and ratio + step >= range[1]:
            step -= 1
        if ratio + step < range[1]:
            ratio += step

    elif bend == False:
        range[1] = ratio
        while step > 0 and ratio - step <= range[0]:
            step -= 1
        if ratio - step > range[0]:
            ratio -= step
    if len(curve1) <= 2:
        p1 = p2 = curve1[0]
    else:
        p1 = curve1[0]
        p2 = ((ratio + 1) * curve1[0][0] - ratio * curve1[2][0],
              (ratio + 1) * curve1[0][1] - ratio * curve1[2][1])
    if len(curve2) <= 2:
        p3 = p4 = curve2[0]
    else:
        p3 = ((ratio + 1) * curve2[0][0] - ratio * curve2[2][0],
              (ratio + 1) * curve2[0][1] - ratio * curve2[2][1])
        p4 = curve2[0]

    segment = bezier_curve_fit(p1, p2, p3, p4, flag)
    print('current range:', range, 'ratio=', ratio)

    return segment, range, ratio


def thinning(segment1, segment2):
    segment2_new = []
    segments = np.array(segment1 + segment2, dtype=np.uint16)
    h, w = np.max(segments, axis=0)

    background = np.zeros((h + 1, w + 1))
    for pos in segment1:
        background[int(pos[0]), int(pos[1])] = 1
    for pos in segment2[::-1]:
        segment2_new.append(pos)
        if search(background, pos):
            break

    return segment2_new


def one2many(curve_one, curve_many_1, curve_many_2, flag, thres=30):
    """
    一对二的情况
    :param curve_one: 一条曲线
    :param curve_many_1: 两条曲线中第一条
    :param curve_many_2: 两条曲线中第二条
    :param flag: 是否细化（去堆积点）
    :param thres: 一对二修复段曲线阈值，超过阈值则将分叉点向两条曲线一侧移动
    :return: 对应两条曲线的修复段及单条曲线延长段，
             curve_many_1-->segment1, curve_many_2-->segment2, extend-->curve_one
    """
    # # 获取两个控制点
    # control1, control2 = get_control_points(curve_one)
    #
    # # print(control1, control2, curve_many_1[0], curve_many_2[0])
    # # 匹配控制点
    # flag2 = partition(curve_many_1[0], curve_many_2[0], control1, control2)
    # (c1, c2) = (control1, control2) if flag2 else (control2, control1)

    c1, c2 = control_points(curve_one, curve_many_1, curve_many_2)
    # 分别修复
    segment1 = restoration_l2p(curve_many_1, curve_one[0], c1, flag)
    segment2 = restoration_l2p(curve_many_2, curve_one[0], c2, flag)
    # print(len(segment1), len(segment2))
    extend = []
    if max(len(segment1), len(segment2)) > thres:
        mid1, mid2 = map(lambda x: len(x) // 2, (segment1, segment2))
        point1, point2 = map(lambda x, idx: x[idx], (segment1, segment2), (mid1, mid2))
        mid_point = (int(0.5 * (point1[0] + point2[0])), int(0.5 * (point1[1] + point2[1])))
        extend = restoration_l2p(curve_one, mid_point, mid_point, flag)
        # curve_one.extend(add_segment)
        c1, c2 = control_points(extend, curve_many_1, curve_many_2)
        segment1 = restoration_l2p(curve_many_1, mid_point, c1, flag)
        segment2 = restoration_l2p(curve_many_2, mid_point, c2, flag)

    segment2 = thinning(segment1, segment2)

    return segment1, segment2 + extend


def draw_curve(pos_list, title='curve', h=400, w=400):
    """
    绘制单条曲线
    :param title:
    :param pos_list: 曲线坐标列表
    :param h:
    :param w:
    :return:
    """
    background = np.ones((h, w), np.uint8) * 255
    for pos in pos_list:
        background[int(pos[0]), int(pos[1])] = 0
    cv2.imshow(title, background)
    cv2.waitKey(0)


if __name__ == '__main__':
    a =1