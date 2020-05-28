# encoding: utf-8
import copy

import numpy as np
import cv2
from Code.description import incomplete_region


####点匹配


def judge_line(end_point, point_line):  # 应该用比例去判断是什么类型
    """
    判断线是x类型还是y类型，用于分类
    """
    x_count = 0
    y_count = 0
    for area in end_point:
        for po in area:
            if len(point_line[po]) > 4:
                flag = abs(po[0] - point_line[po][4][0]) > abs(po[1] - point_line[po][4][1]) + 1
            else:  # 线太短
                flag = abs(po[0] - point_line[po][-1][0]) > abs(po[1] - point_line[po][-1][1])
            if flag:
                y_count += 1
            else:
                x_count += 1

    if x_count / (x_count + y_count) > 0.6:
        return True  # ｘ类型
    else:
        return False  # ｙ类型　数值


def judge_derivative(i, line):  # 判断一个点是不是转折点
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    for j in range(i - 4, i):
        x_1.append(line[j][1])
        y_1.append(line[j][0])
    for j in range(i + 1, i + 5):
        x_2.append(line[j][1])
        y_2.append(line[j][0])

    if len(set(x_1)) == 1 or len(set(x_2)) == 1:
        der_1 = np.polyfit(y_1, x_1, 1)[0]
        der_2 = np.polyfit(y_2, x_2, 1)[0]
    else:
        der_1 = np.polyfit(x_1, y_1, 1)[0]
        der_2 = np.polyfit(x_2, y_2, 1)[0]

    angle = abs((der_1 - der_2) / (1 + der_1 * der_2))
    if angle > 0.35:
        return True, angle  # 转折点
    else:
        return False, angle  # 平滑点


def turning_point(line):  # 用一个点的左导数和右导数区别值
    select_line = line
    turn_point = []
    der_diff_dict = {}
    for i in range(0, len(select_line), 2):
        if i > 3 and i < len(select_line) - 4:
            flag, der_diff = judge_derivative(i, select_line)
            if flag:
                turn_point.append(select_line[i])
                der_diff_dict[select_line[i]] = der_diff

    return turn_point, der_diff_dict


def cal_dis(po1, po2):
    return ((po1[0] - po2[0]) ** 2 + (po1[1] - po2[1]) ** 2) ** 0.5


def edge_judge_connect(first_line, domain_line, candidate_line1, flag):  # 判断边缘断点是否有潜在连接点
    """
    :param first_line: 断点所在的线
    :param domain_line: 残缺区域平行线,已经确定了key
    :param candidate_line1: 对面备选第一条线,和潜在点是竞争关系
    :param flag: x或y类型
    :return:
    """
    candidate_point = []  # 储存断裂点
    dict = {}  # 存储断裂点的斜率差值
    for line in domain_line:
        turning_point1, der_diff_dict = turning_point(line)
        candidate_point = candidate_point + turning_point1
        dict.update(der_diff_dict)
    ###方案一
    candidate_point_list = \
        sorted(candidate_point,
               key=lambda x: cal_dis(x, candidate_line1[0]))[:4]
    new_candidate_point_list = candidate_point_list.copy()
    dis = cal_dis(first_line[0], candidate_line1[0])
    for po in new_candidate_point_list:  # 删除掉距离太远的点
        distance_judge = cal_dis(po, candidate_line1[0])
        if distance_judge > 13:
            candidate_point_list.remove(po)

    if len(candidate_point_list) == 0:
        return False, first_line, candidate_line1
    candidate_point = sorted(candidate_point_list, key=lambda x: dict[x])[-1]

    for line in domain_line:
        if candidate_point in line:
            p = line.index(candidate_point)
            p1 = line[:p + 1]
            p1.reverse()
            p2 = line[p:]
            length = 3
            if len(first_line) < 4:
                length = len(first_line) - 1
            length1 = 3
            if len(p1) < 4:
                length1 = len(p1) - 1
            if flag:
                if (first_line[0][1] - first_line[length][1]) * (p1[0][1] - p1[length1][1]) < 0:
                    candidate_line = p1  # 放入edge_point_line 这是一对一连接需要的线
                    candidate_line_1 = p2  # 放入point_line 这是检测是否密集
                else:
                    candidate_line = p2
                    candidate_line_1 = p1
            else:
                if (first_line[0][0] - first_line[length][0]) * (p1[0][0] - p1[length1][0]) < 0:
                    candidate_line = p1
                    candidate_line_1 = p2
                else:
                    candidate_line = p2
                    candidate_line_1 = p1
    point_line = {}
    point_line[first_line[0]] = first_line
    point_line[candidate_line1[0]] = candidate_line1
    point_line[candidate_point] = candidate_line

    score1 = match_judge(first_line[0], candidate_point, flag, point_line)+0.4 #惩罚项
    score2 = match_judge(first_line[0], candidate_line1[0], flag, point_line)
    if score1 < score2:
        return True, candidate_line, candidate_line_1  # 与潜在点连接
    else:
        return False, candidate_line, candidate_line_1  # 不与潜在点连接


def angle(vector1, vector2):
    # 计算两个向量的夹角
    result1 = 0
    result2 = 0
    result3 = 0
    for i in range(2):
        result1 += vector1[i] * vector2[i]  # sum(X*Y)
        result2 += vector1[i] ** 2  # sum(X*X)
        result3 += vector2[i] ** 2  # sum(Y*Y)
    cos_angle = result1 / (result2 * result3) ** 0.5
    return np.arccos(cos_angle) * 180 / np.pi


def match_judge(point_1, point_2, flag, point_line):
    # flag为区域是x类型还是y类型
    length1 = 5
    if len(point_line[point_1]) < 6:
        length1 = len(point_line[point_1]) - 1
    length2 = 5
    if len(point_line[point_2]) < 6:
        length2 = len(point_line[point_2]) - 1
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    for po in point_line[point_1][:length1 + 1]:
        x_1.append(po[1])
        y_1.append(po[0])
    for po in point_line[point_2][:length2 + 1]:
        x_2.append(po[1])
        y_2.append(po[0])
    if flag:  # ｘ类型
        slope_1 = np.polyfit(x_1, y_1, 1)[0]
        slope_2 = np.polyfit(x_2, y_2, 1)[0]
        # slope = (point_1[0] - point_2[0]) / (point_1[1] - point_2[1])
        vector_1 = (1, slope_1) if point_line[point_1][length1][1] > point_1[1] else (-1, -slope_1)  # 向量都是由断点开始,指向另一边
        vector_2 = (1, slope_2) if point_line[point_2][length2][1] > point_2[1] else (-1, -slope_2)
        line_vector1 = (point_2[1] - point_1[1], point_2[0] - point_1[0])
        line_vector2 = (point_1[1] - point_2[1], point_1[0] - point_2[0])
    else:  # ｙ类型
        slope_1 = np.polyfit(y_1, x_1, 1)[0]
        slope_2 = np.polyfit(y_2, x_2, 1)[0]
        # slope = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
        vector_1 = (1, slope_1) if point_line[point_1][length1][0] > point_1[0] else (-1, -slope_1)  # 向量都是由断点开始,指向另一边
        vector_2 = (1, slope_2) if point_line[point_2][length2][0] > point_2[0] else (-1, -slope_2)
        line_vector1 = (point_2[0] - point_1[0], point_2[1] - point_1[1])
        line_vector2 = (point_1[0] - point_2[0], point_1[1] - point_2[1])
    a1 = angle(vector_1, line_vector1)
    a2 = angle(vector_2, line_vector2)
    if angle(vector_1, vector_2) > 165:  # 向量夹角超过165°,为直线连接
        return 360 - angle(vector_1, line_vector1) - angle(vector_2, line_vector2)
    else:  # 曲线
        return max(180 - angle(vector_1, line_vector1), 180 - angle(vector_2, line_vector2))  # 越小越好


def contours(gray):
    # 转换为二值图像
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 2次腐蚀,3次膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.erode(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=3)

    cloneImg, contours, heriachy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_img = np.ones(binary.shape, dtype=np.uint8) * 255  # 创建白色背景
    ###　凸包轮廓检测，保证边缘残缺在里面
    max_loc = 0
    for i, contour in enumerate(contours):
        if len(contour) >= len(contours[max_loc]):
            max_loc = i
    cv2.drawContours(new_img, contours, max_loc, 0, cv2.FILLED)  # 对临时图填充黑色

    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #####当两个轮廓特别近的时候,修改膨胀系数
    new_img = cv2.dilate(new_img, new_kernel, iterations=2)  # 对黑色背景腐蚀，使白色扩大
    # cv2.imshow("erode", new_img)
    # cv2.waitKey(0)
    return new_img


def find_side_point(point, domain_line1, temp_img):
    # 寻找断点最近的那条线的终点
    min_dis = float("inf")
    min_line_loc = 0
    for i, line in enumerate(domain_line1):
        for po in line:
            if cal_dis(point, po) <= min_dis:
                min_dis = cal_dis(point, po)
                min_line_loc = i
    if temp_img[domain_line1[min_line_loc][0][0], domain_line1[min_line_loc][0][1]] == 255:
        if temp_img[domain_line1[min_line_loc][-1][0], domain_line1[min_line_loc][-1][1]] == 255 and cal_dis(point,
                                                                                                             domain_line1[
                                                                                                                 min_line_loc][
                                                                                                                 0]) > cal_dis(
            point, domain_line1[min_line_loc][-1]):
            return domain_line1[min_line_loc][-1], min_dis
        else:
            return domain_line1[min_line_loc][0], min_dis
    else:
        if temp_img[domain_line1[min_line_loc][-1][0], domain_line1[min_line_loc][-1][1]] == 0 and cal_dis(point,
                                                                                                           domain_line1[
                                                                                                               min_line_loc][
                                                                                                               0]) < cal_dis(
            point, domain_line1[min_line_loc][-1]):
            return domain_line1[min_line_loc][0], min_dis
        else:
            return domain_line1[min_line_loc][-1], min_dis


def point_match(classify_point, point_line, domain_line, img):
    match_point = {}
    dict_match_point = {}
    side_match_point = {}
    edge_point_line = {}  # 储存边缘连接的线
    for key in classify_point:
        count = 0  # 计数
        match_point[key] = []
        if not classify_point[key][0] or not classify_point[key][1]:  # 对某个区域某边没有断点
            side_breakpoint = classify_point[key][0] if len(classify_point[key][0]) >= len(classify_point[key][1]) else \
                classify_point[key][1]
            if len(side_breakpoint) > 1:  ##指纹边界残缺延长
                side_flag = False
                temp_img = contours(img)
                if temp_img[side_breakpoint[0][0], side_breakpoint[0][1]] == 255:
                    side_flag = True
                if side_flag:  # 是指纹边界残缺,点全在一边
                    if len(domain_line[key]) > 0:
                        side_point1, min_dis1 = find_side_point(side_breakpoint[0], domain_line[key], temp_img)
                        side_point2, min_dis2 = find_side_point(side_breakpoint[-1], domain_line[key], temp_img)
                        if side_point1 == side_point2:
                            if min_dis1 < min_dis2:
                                side_point2 = side_breakpoint[-1]
                                side_breakpoint.remove(side_point2)
                            else:
                                side_point1 = side_breakpoint[0]
                                side_breakpoint.remove(side_point1)
                    else:
                        side_point1 = side_breakpoint[0]
                        side_point2 = side_breakpoint[-1]
                        side_breakpoint.remove(side_point1)
                        side_breakpoint.remove(side_point2)
                    if side_point1 != side_point2:  # 避免都是一条线的点
                        for j in range(len(side_breakpoint)):
                            possible_point = (round(
                                (side_point2[0] - side_point1[0]) * (j + 1) / (len(side_breakpoint) + 1) + side_point1[
                                    0]), round(
                                (side_point2[1] - side_point1[1]) * (j + 1) / (len(side_breakpoint) + 1) + side_point1[
                                    1]))
                            side_match_point[side_breakpoint[j]] = [(int(possible_point[0]), int(possible_point[1]))]
                else:  # 内部残缺,点全在一边.
                    point_nums = len(side_breakpoint)
                    for i in range(int(point_nums / 2)):
                        match_point[key].append([side_breakpoint[i], side_breakpoint[point_nums - i - 1]])
                        dict_match_point[side_breakpoint[i]] = [side_breakpoint[point_nums - i - 1]]

            continue
        flag = judge_line(classify_point[key], point_line)
        if len(classify_point[key][0]) > len(classify_point[key][1]):
            classify_point[key][0], classify_point[key][1] = classify_point[key][1], classify_point[key][0]
        data_point_1 = classify_point[key][1].copy()
        data_point_0 = classify_point[key][0].copy()
        temp_data_point_1 = classify_point[key][1].copy()
        signal = 1
        while len(data_point_0) != 0:
            if len(data_point_0) == len(data_point_1):
                for i in range(len(data_point_0)):
                    match_point[key].append([data_point_0[i], data_point_1[i]])
                    dict_match_point[data_point_0[i]] = [data_point_1[i]]
                break
            ###判断边缘潜在连接
            if count == 0:
                edge_data_point_sign = [0, -1]
                for edge_sign in edge_data_point_sign:
                    flag1, candidate_line, candidate_line_1 = edge_judge_connect(point_line[data_point_1[edge_sign]],
                                                                                 domain_line[key],
                                                                                 point_line[data_point_0[edge_sign]],
                                                                                 flag)
                    flag2 = cal_dis(candidate_line[0], data_point_1[edge_sign]) < cal_dis(candidate_line[0],
                                                                                          data_point_1[-1 - edge_sign])
                    if flag1 and flag2:  # 与潜在断点链接,怕连到另一边去
                        if candidate_line[0] not in point_line:
                            point_line[candidate_line[0]] = candidate_line
                        match_point[key].append([candidate_line[0], data_point_1[edge_sign]])
                        # dict_match_point[candidate_line[0]] = [data_point_1[edge_sign]]
                        side_match_point[data_point_1[edge_sign]] = [candidate_line[0]]
                        edge_point_line[candidate_line[0]] = candidate_line_1  # 存储判断是否紧密的线
                        del data_point_1[edge_sign]
            ###########
            if len(data_point_1) < len(data_point_0):  # 由于边缘修补的可能性,当1比0少的时候,则两个互换
                data_point_1, data_point_0 = data_point_0, data_point_1
            if len(data_point_0) == len(data_point_1):
                for i in range(len(data_point_0)):
                    match_point[key].append([data_point_0[i], data_point_1[i]])
                    dict_match_point[data_point_0[i]] = [data_point_1[i]]
                break
            ################中心区修复

            #####当两边点数目相差悬殊时
            side_breakpoint = data_point_1.copy()
            while len(data_point_1) - len(data_point_0) > 1:
                if len(data_point_0) == 0:
                    break
                minus_value = len(data_point_1) - len(data_point_0)
                po0 = data_point_0[0]
                min_score = match_judge(po0, data_point_1[0], flag, point_line)
                min_loc = 0
                for n in range(minus_value + 1):
                    temp_score = match_judge(po0, data_point_1[n], flag, point_line)
                    if temp_score <= min_score:
                        min_score = temp_score
                        min_loc = n
                if min_loc > 0:
                    score_1 = match_judge(po0, data_point_1[min_loc - 1], flag, point_line)
                else:
                    score_1 = float("inf")
                if min_loc < minus_value:
                    score_2 = match_judge(po0, data_point_1[min_loc + 1], flag, point_line)
                else:
                    score_2 = float("inf")
                if score_1 > score_2:
                    match_point[key].append([po0, data_point_1[min_loc]])
                    if abs(score_2 - min_score) < 12:
                        match_point[key].append([po0, data_point_1[min_loc + 1]])
                        dict_match_point[po0] = [data_point_1[min_loc], data_point_1[min_loc + 1]]
                        side_breakpoint.remove(data_point_1[min_loc])
                        side_breakpoint.remove(data_point_1[min_loc + 1])
                        del data_point_1[0]
                    else:
                        dict_match_point[po0] = [data_point_1[min_loc]]
                        side_breakpoint.remove(data_point_1[min_loc])
                else:
                    match_point[key].append([po0, data_point_1[min_loc]])
                    if abs(score_1 - min_score) < 12:
                        match_point[key].append([po0, data_point_1[min_loc - 1]])
                        dict_match_point[po0] = [data_point_1[min_loc], data_point_1[min_loc - 1]]
                        side_breakpoint.remove(data_point_1[min_loc])
                        side_breakpoint.remove(data_point_1[min_loc - 1])
                    else:
                        dict_match_point[po0] = [data_point_1[min_loc]]
                        side_breakpoint.remove(data_point_1[min_loc])
                del data_point_0[0]
                for i in range(min_loc + 1):
                    del data_point_1[0]
            #####指纹边界残缺延长
            for po1 in data_point_1:
                side_breakpoint.remove(po1)
            if len(side_breakpoint) > 1:  ##指纹边缘
                side_flag = False
                temp_img = contours(img)
                if temp_img[side_breakpoint[0][0], side_breakpoint[0][1]] == 255:
                    side_flag = True
                if side_flag:
                    if len(domain_line[key]) > 0:
                        side_point1, min_dis1 = find_side_point(temp_data_point_1[0], domain_line[key],
                                                                temp_img)  # 采用的是最开始旁边的点,而不是side_breakpoint的首尾点
                        side_point2, min_dis2 = find_side_point(temp_data_point_1[-1], domain_line[key], temp_img)
                        if side_point1 == side_point2:  ##有可能最旁边的边缘点被识别成了断点,导致其最旁边没有domain_line
                            if min_dis1 < min_dis2:
                                side_point2 = temp_data_point_1[-1]
                                if side_point2 in side_breakpoint:
                                    side_breakpoint.remove(side_point2)
                                temp_data_point_1.remove(side_point2)
                            else:
                                side_point1 = temp_data_point_1[0]
                                if side_point1 in side_breakpoint:
                                    side_breakpoint.remove(side_point1)
                                temp_data_point_1.remove(side_point1)
                            # segment_nums = len(temp_data_point_1)
                    else:
                        side_point1 = temp_data_point_1[0]
                        side_point2 = temp_data_point_1[-1]
                        if side_point1 in side_breakpoint:
                            side_breakpoint.remove(side_point1)
                        if side_point2 in side_breakpoint:
                            side_breakpoint.remove(side_point2)
                        temp_data_point_1.remove(side_point1)
                        temp_data_point_1.remove(side_point2)
                        # segment_nums = len(temp_data_point_1) - 1
                    if side_point1 != side_point2:
                        segment_nums = len(temp_data_point_1) + 1
                        for side_po in side_breakpoint:
                            j = temp_data_point_1.index(side_po)  # 因为有部分是已经连接了,所以要考虑他们才能确定点的位置
                            possible_point = (round(
                                (side_point2[0] - side_point1[0]) * (j + 1) / segment_nums + side_point1[0]), round(
                                (side_point2[1] - side_point1[1]) * (j + 1) / segment_nums + side_point1[1]))
                            side_match_point[side_po] = [(int(possible_point[0]), int(possible_point[1]))]

            if len(data_point_0) == 0:
                break
            if len(data_point_0) == len(data_point_1):
                for i in range(len(data_point_0)):
                    match_point[key].append([data_point_0[i], data_point_1[i]])
                    dict_match_point[data_point_0[i]] = [data_point_1[i]]
                break

            #######当点数目相差不大时
            sign = 0 if signal > 0 else -1
            score1 = match_judge(data_point_0[sign], data_point_1[sign], flag, point_line)
            score2 = match_judge(data_point_0[sign], data_point_1[sign + signal], flag, point_line)
            flag2 = True  # 判断第二根该不该连的一个子条件
            if len(data_point_0) > 1:
                score3 = match_judge(data_point_0[sign + signal], data_point_1[sign + signal], flag,
                                     point_line)  # 两边的第二条匹配,这个是看多的那组第二条更适合那个
                if score2 > score3:
                    flag2 = False
            if abs(score1 - score2) < 12 and flag2:
                match_point[key].append([data_point_0[sign], data_point_1[sign]])
                match_point[key].append([data_point_0[sign], data_point_1[sign + signal]])
                dict_match_point[data_point_0[sign]] = [data_point_1[sign], data_point_1[sign + signal]]
                del data_point_0[sign], data_point_1[sign], data_point_1[sign]  # 因为删掉了第一个，第二个就变成了第一个
            elif score1 - score2 >= 12:
                match_point[key].append([data_point_0[sign], data_point_1[sign + signal]])
                dict_match_point[data_point_0[sign]] = [data_point_1[sign + signal]]
                del data_point_0[sign], data_point_1[sign], data_point_1[sign]  # 匹配的是第二个根，所以外面的那根不可能有人匹配
            else:
                match_point[key].append([data_point_0[sign], data_point_1[sign]])
                dict_match_point[data_point_0[sign]] = [data_point_1[sign]]
                del data_point_0[sign], data_point_1[sign]

            signal = -signal
            count += 1
    return match_point, dict_match_point, point_line, side_match_point, edge_point_line


def draw(img, match_point):  # 对配对点用颜色标识
    h, w = img.shape
    new_img = np.zeros((h, w, 3), np.uint8)
    new_img[:, :, 0] = new_img[:, :, 1] = new_img[:, :, 2] = img
    colors = [[252, 230, 202], [255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [138, 43, 226], [170, 219, 245],
              [150, 0, 100], [252, 230, 202], [255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255]]
    # 灰色，蓝色，绿色,红色,黄色，土耳其蓝,深红，暗黄，
    for key in match_point:
        temp = match_point[key]
        num = 0
        while len(temp) != 0:
            if len(temp) > 1 and temp[0][0] == temp[1][0]:
                new_img[temp[0][0][0], temp[0][0][1]] = new_img[temp[0][1][0], temp[0][1][1]] = new_img[
                    temp[1][1][0], temp[1][1][1]] = colors[num]
                del temp[0], temp[0]
            else:
                new_img[temp[0][0][0], temp[0][0][1]] = new_img[temp[0][1][0], temp[0][1][1]] = colors[num]
                del temp[0]
            num += 1
    cv2.imshow("1", img)
    cv2.imshow("final", new_img)
    cv2.waitKey(0)


def final(img):
    result, classify_point, point_line, domain_line = incomplete_region.getCoordinate(img)
    match_point, dict, point_line, side_match_point, edge_point_line = point_match(classify_point, point_line,
                                                                                   domain_line, result)
    # draw(result, match_point)  # 画图
    return dict, point_line, side_match_point, edge_point_line

