# encoding: utf-8
import numpy as np
import math
from Code.reconstruction.ridge_restoration import bezier_curve_fit, draw_curve, fix_control_points, judge_intersection, one2one
import copy

def rotate_point(p1, p2, point, theta):
    shifted_point = (point[0] - p1[0], point[1] - p1[1])

    v = (p2[0] - p1[0], p2[1] - p1[1])
    cos_v = v[0] / math.sqrt(v[0] ** 2 + v[1] ** 2)
    sin_v = v[1] / math.sqrt(v[0] ** 2 + v[1] ** 2)
    # 变换坐标轴之后的point
    new_point = (shifted_point[0] * cos_v + shifted_point[1] * sin_v,
                 -1 * shifted_point[0] * sin_v + shifted_point[1] * cos_v)
    # 判断象限
    quadrant = 1
    if new_point[0] < 0 and new_point[1] >= 0:
        quadrant = 2
    elif new_point[0] < 0 and new_point[1] < 0:
        quadrant = 3
    elif new_point[0] >= 0 and new_point[1] < 0:
        quadrant = 4

    rotated_point = (math.fabs(new_point[0]) * math.cos(theta) - math.fabs(new_point[1]) * math.sin(theta),
                     math.fabs(new_point[1]) * math.cos(theta) + math.fabs(new_point[0]) * math.sin(theta))

    if quadrant == 2:
        rotated_point = (-rotated_point[0], rotated_point[1])
    elif quadrant == 3:
        rotated_point = (-rotated_point[0], -rotated_point[1])
    elif quadrant == 4:
        rotated_point = (rotated_point[0], -rotated_point[1])

    rotated_point = (int(rotated_point[0] * cos_v - rotated_point[1] * sin_v + p1[0]),
                     int(rotated_point[1] * cos_v + rotated_point[0] * sin_v + p1[1]))

    return rotated_point


def one2one_adjust(curve1, curve2, flag, neighbor=None, bend_range1=None, bend_range2=None,
                   cur_point1=None, cur_point2=None, bend_flag=None, theta1=None, theta2=None, theta_flag=False,
                   h=400, w=400):
    ans = True
    segment = None

    new_bend_range1 = copy.copy(bend_range1)
    new_bend_range2 = copy.copy(bend_range2)
    new_cur_point1 = copy.copy(cur_point1)
    new_cur_point2 = copy.copy(cur_point2)


    # 弯曲程度增大

    if theta_flag:
        # rotate control points

        cur_point1_p = rotate_point(curve1[0], curve2[0], new_cur_point1, theta1)
        cur_point2_p = rotate_point(curve2[0], curve1[0], new_cur_point2, theta2)

        new_bend_range1[0] = curve1[0]
        new_bend_range2[0] = curve2[0]
        intersection = None
        try:
            intersection = line_intersection((curve1[0], cur_point1_p),
                                             (curve2[0], cur_point2_p))
        except Exception as e:
            # print(e)
            ans = False
        else:
            new_bend_range1[1] = intersection
            new_bend_range2[1] = intersection

    elif bend_flag == True:
        if math.fabs(new_bend_range1[1][0] - new_cur_point1[0]) <= 1 or \
                math.fabs(new_bend_range1[1][1] - new_cur_point1[1]) <= 1 or \
                math.fabs(new_bend_range2[1][0] - new_cur_point2[0]) <= 1 or \
                math.fabs(new_bend_range2[1][1] - new_cur_point2[1]) <= 1:

            # print("At current theta, curvature cannot continue to increase")
            ans = False


        else:
            new_bend_range1[0] = new_cur_point1
            new_bend_range2[0] = new_cur_point2

    elif bend_flag == False:
        if math.fabs(new_bend_range1[0][0] - new_cur_point1[0]) <= 1 and \
                math.fabs(new_bend_range1[0][1] - new_cur_point1[1]) <= 1 and \
                math.fabs(new_bend_range2[0][0] - new_cur_point2[0]) <= 1 and \
                math.fabs(new_bend_range2[0][1] - new_cur_point2[1]) <= 1:
            # print("At current theta, curvature cannot continue to decrease")
            ans = False

        else:
            new_bend_range1[1] = new_cur_point1
            new_bend_range2[1] = new_cur_point2

    if ans:
        new_cur_point1 = ((new_bend_range1[0][0] + new_bend_range1[1][0]) // 2,
                          (new_bend_range1[0][1] + new_bend_range1[1][1]) // 2)
        new_cur_point2 = ((new_bend_range2[0][0] + new_bend_range2[1][0]) // 2,
                          (new_bend_range2[0][1] + new_bend_range2[1][1]) // 2)

        segment = bezier_curve_fit(curve1[0], new_cur_point1, new_cur_point2, curve2[0], flag)

        flag_cross = False
        if neighbor and len(set(segment) & set(neighbor)) > 0:
            flag_cross = True
        else:
            xx = [pos[0] for pos in segment]
            yy = [pos[1] for pos in segment]
            if min(xx) < 0 or max(xx) >= h or min(yy) < 0 or max(yy) > w:
                flag_cross = True

        if flag_cross:
            # print('the restoration segment cross with the adjacent ridge or boundary')
            return one2one_adjust(curve1, curve2, flag, neighbor, bend_flag=False,
                                  bend_range1=new_bend_range1, bend_range2=new_bend_range2,
                                  cur_point1=new_cur_point1, cur_point2=new_cur_point2,
                                  theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)

    return ans, segment, new_bend_range1, new_bend_range2, new_cur_point1, new_cur_point2


def one2one_candidates(curve1, curve2, neighbor=None, h=400, w=400):
    candidates = []
    info = []

    f = True
    symbol = True

    # initialize
    bend_range1 = [None, None]
    bend_range2 = [None, None]

    pos1 = 2
    pos2 = 2
    t = 1
    intersection = None
    while True:

        temp1 = None
        if len(curve1) <= pos1:
            temp1 = curve1[-1]
        else:
            temp1 = curve1[pos1]
        bend_range1[0] = (2 * curve1[0][0] - temp1[0],
                          2 * curve1[0][1] - temp1[1])
        temp2 = None
        if len(curve2) <= pos2:
            temp2 = curve2[-1]
        else:
            temp2 = curve2[pos2]
        bend_range2[0] = (2 * curve2[0][0] - temp2[0],
                          2 * curve2[0][1] - temp2[1])
        cur_point1 = (5 * curve1[0][0] - 4 * temp1[0],
                      5 * curve1[0][1] - 4 * temp1[1])
        cur_point2 = (5 * curve2[0][0] - 4 * temp2[0],
                      5 * curve2[0][1] - 4 * temp2[1])

        if not judge_intersection(curve1[0], cur_point1, curve2[0], cur_point2):
            cur_point1, cur_point2, symbol = fix_control_points(curve1, curve2, cur_point1, cur_point2)
        try:
            intersection = line_intersection((curve1[0], cur_point1),
                                             (curve2[0], cur_point2))
        except Exception as e:
            # print(e)
            # print((curve1[0], cur_point1), (curve2[0], cur_point2))
            if pos1 >= len(curve1) and pos2 >= len(curve2):
                f = False
                break
            if t % 2:
                pos1 += 1
            else:
                pos2 += 1
            t += 1
            continue
        else:
            break

    if not f:
        candidates.append(one2one(curve1, curve2, flag=False))
        info.append("not adjust")
        return candidates, info
    # if not ans:
    #     return None, None

    bend_range1[1] = intersection
    bend_range2[1] = intersection

    v1 = (curve2[0][0] - curve1[0][0], curve2[0][1] - curve1[0][1])
    v2 = (cur_point1[0] - curve1[0][0], cur_point1[1] - curve1[0][1])
    v3 = (-1 * v1[0], -1 * v1[1])
    v4 = (cur_point2[0] - curve2[0][0], cur_point2[1] - curve2[0][1])

    theta1 = (0.5 * math.pi -
              math.acos(round(np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2))), 3))) / 5
    theta2 = (0.5 * math.pi -
              math.acos(round(np.dot(v3, v4) / (np.linalg.norm(v3) * (np.linalg.norm(v4))), 3))) / 5

    if not f:
        candidates.append(one2one(curve1, curve2, flag=False))

    '''
    at 1-th theta
    '''
    ans, segment, bend_range1, bend_range2, cur_point1, cur_point2 = \
        one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=None,
                       bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                       cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
    if ans:
        info.append(['1th'])
        candidates.append(segment)

    # decrease curvature
    ans, segment, bend_range1_temp, bend_range2_temp, cur_point1_temp, cur_point2_temp = \
        one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=False,
                       bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                       cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
    if ans:
        info.append(['1th, d1'])
        candidates.append(segment)

        # continue to decrease curvature
        ans, segment, _, _, _, _ = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=False,
                           bend_range1=bend_range1_temp, bend_range2=bend_range2_temp, cur_point1=cur_point1_temp,
                           cur_point2=cur_point2_temp, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
        if ans:
            info.append(['1th, d2'])
            candidates.append(segment)

    # increase curvature
    ans, segment, bend_range1_temp, bend_range2_temp, cur_point1_temp, cur_point2_temp = \
        one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=True,
                       bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                       cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
    if ans:
        info.append(['1th, i1'])
        candidates.append(segment)

        # continue to increase curvature
        ans, segment, _, _, _, _ = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=True,
                           bend_range1=bend_range1_temp, bend_range2=bend_range2_temp, cur_point1=cur_point1_temp,
                           cur_point2=cur_point2_temp, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
        if ans:
            info.append(['1th, i2'])

            candidates.append(segment)

    if symbol:
        '''
        at 2-th theta
        '''
        ans, segment, bend_range1, bend_range2, cur_point1, cur_point2 = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=None,
                           bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                           cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=True, h=h, w=w)

        if ans:
            info.append(['2th'])
            candidates.append(segment)

        # decrease curvature
        ans, segment, bend_range1_temp, bend_range2_temp, cur_point1_temp, cur_point2_temp = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=False,
                           bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                           cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
        if ans:
            info.append(['2th, d1'])
            candidates.append(segment)

            # continue to decrease curvature
            ans, segment, _, _, _, _ = \
                one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=False,
                               bend_range1=bend_range1_temp, bend_range2=bend_range2_temp, cur_point1=cur_point1_temp,
                               cur_point2=cur_point2_temp, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
            if ans:
                info.append(['2th, d2'])
                candidates.append(segment)

        # increase curvature
        ans, segment, bend_range1_temp, bend_range2_temp, cur_point1_temp, cur_point2_temp = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=True,
                           bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                           cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
        if ans:
            info.append(['2th, i1'])
            candidates.append(segment)

            # continue to increase curvature
            ans, segment, _, _, _, _ = \
                one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=True,
                               bend_range1=bend_range1_temp, bend_range2=bend_range2_temp, cur_point1=cur_point1_temp,
                               cur_point2=cur_point2_temp, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
            if ans:
                info.append(['2th, i2'])
                candidates.append(segment)

        '''
        at 3-th theta
        '''
        ans, segment, bend_range1, bend_range2, cur_point1, cur_point2 = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=None,
                           bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                           cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=True, h=h, w=w)

        if ans:
            info.append(['3th'])
            candidates.append(segment)
        # decrease curvature
        ans, segment, bend_range1_temp, bend_range2_temp, cur_point1_temp, cur_point2_temp = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=False,
                           bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                           cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
        if ans:
            info.append(['3th, d1'])
            candidates.append(segment)

            # continue to decrease curvature
            ans, segment, _, _, _, _ = \
                one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=False,
                               bend_range1=bend_range1_temp, bend_range2=bend_range2_temp, cur_point1=cur_point1_temp,
                               cur_point2=cur_point2_temp, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
            if ans:
                info.append(['3th, d2'])
                candidates.append(segment)

        # increase curvature
        ans, segment, bend_range1_temp, bend_range2_temp, cur_point1_temp, cur_point2_temp = \
            one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=True,
                           bend_range1=bend_range1, bend_range2=bend_range2, cur_point1=cur_point1,
                           cur_point2=cur_point2, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
        if ans:
            info.append(['3th, i1'])
            candidates.append(segment)

            # continue to increase curvature
            ans, segment, _, _, _, _ = \
                one2one_adjust(curve1=curve1, curve2=curve2, flag=False, neighbor=neighbor, bend_flag=True,
                               bend_range1=bend_range1_temp, bend_range2=bend_range2_temp, cur_point1=cur_point1_temp,
                               cur_point2=cur_point2_temp, theta1=theta1, theta2=theta2, theta_flag=False, h=h, w=w)
            if ans:
                info.append(['3th, i2'])
                candidates.append(segment)

    return candidates, info


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


if __name__ == '__main__':

    adjacent = [(247, 24), (246, 25), (245, 26), (244, 27), (244, 28), (243, 29), (242, 30), (242, 31), (241, 32), (240, 33), (240, 34), (239, 35), (238, 36), (238, 37), (237, 38), (237, 39), (236, 40), (236, 41), (235, 42), (235, 43), (234, 44), (234, 45), (234, 46), (233, 47), (233, 48), (232, 49), (232, 50), (231, 51), (231, 52), (231, 53), (230, 54), (230, 55), (229, 56), (229, 57), (228, 58), (228, 59), (227, 60), (227, 61), (226, 62), (226, 63), (225, 64), (225, 65), (224, 66), (224, 67), (223, 68), (223, 69), (222, 70), (222, 71), (221, 72), (221, 73), (221, 74), (220, 75), (220, 76), (219, 77), (219, 78), (219, 79), (218, 80), (218, 81), (218, 82), (217, 83), (217, 84), (217, 85), (216, 86), (216, 87), (215, 88), (215, 89), (214, 90), (214, 91), (213, 92), (213, 93), (213, 94), (212, 95), (212, 96), (211, 97), (211, 98), (211, 99), (210, 100), (210, 101), (210, 102), (210, 103), (210, 104), (209, 105), (209, 106), (209, 107), (209, 108), (209, 109), (209, 110), (209, 111), (209, 112), (209, 113), (209, 114), (209, 115), (209, 116), (209, 117), (209, 118), (209, 119), (209, 120), (209, 121), (209, 122), (209, 123), (209, 124), (209, 125), (210, 126), (210, 127), (210, 128), (210, 129), (211, 130), (211, 131), (211, 132), (212, 133), (212, 134), (212, 135), (213, 136), (213, 137), (213, 138), (213, 139), (214, 140), (214, 141), (214, 142), (214, 143), (214, 144), (214, 145), (214, 146), (214, 147), (214, 148), (214, 149), (215, 150), (215, 151), (215, 152), (215, 153), (216, 154), (216, 155), (216, 156), (217, 157), (217, 158), (217, 159), (218, 160), (218, 161), (219, 162), (219, 163), (219, 164), (220, 165), (220, 166), (221, 167), (221, 168), (221, 169), (222, 170), (222, 171), (223, 172), (223, 173), (224, 174), (224, 175), (225, 176), (225, 177), (226, 178), (227, 179), (227, 180), (228, 181), (229, 182), (229, 183), (230, 184), (231, 185), (232, 186), (233, 187), (233, 188), (234, 189), (235, 190), (236, 191), (236, 192), (237, 193), (238, 194), (238, 195), (239, 196), (239, 197), (240, 198), (240, 199), (240, 200), (240, 201), (240, 202), (240, 203), (241, 204), (242, 205), (242, 206), (243, 207), (243, 208), (244, 209), (244, 210), (245, 211), (246, 212), (246, 213), (247, 214), (247, 215), (248, 216), (248, 217), (249, 218), (249, 219), (250, 220), (250, 221), (251, 222), (251, 223), (252, 224), (252, 225), (253, 226), (254, 227), (254, 228), (255, 229), (255, 230), (256, 231), (257, 231)]


    curve1 = [(202, 180), (202, 179), (201, 178), (200, 177), (200, 176), (200, 175), (199, 174), (199, 173), (198, 172), (198, 171), (197, 170), (197, 169), (196, 168), (196, 167), (196, 166), (195, 165), (195, 164), (194, 163), (194, 162), (193, 161), (193, 160), (192, 159), (192, 158), (191, 157), (191, 156), (190, 155), (190, 154), (190, 153), (190, 152), (190, 151), (191, 150), (191, 149), (191, 148), (191, 147), (190, 146), (190, 145), (190, 144), (190, 143), (190, 142), (190, 141), (190, 140), (190, 139), (190, 138), (190, 137)]


    curve2 = [(223, 211), (224, 212), (224, 213), (225, 214), (225, 215), (226, 216), (227, 217), (227, 218), (228, 219), (229, 220), (229, 221), (230, 222), (231, 223), (232, 224), (233, 225), (233, 226), (234, 227), (235, 228), (236, 229), (237, 230), (238, 231), (239, 231), (240, 232), (241, 233), (242, 233)]


    draw_curve(curve1 + curve2 + adjacent)
    candidates, info = one2one_candidates(curve1, curve2, neighbor=adjacent, h=400, w=400)
    print(candidates)
    if candidates:
        for i in range(len(candidates)):
            print(info[i])
            draw_curve(curve1 + curve2 + candidates[i] + adjacent)