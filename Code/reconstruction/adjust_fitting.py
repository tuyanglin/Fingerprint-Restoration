# encoding: utf-8
'''
description : Fingerprint repair script
'''

from Code.reconstruction.point_match import final
from Code.reconstruction.ridge_restoration import one2one, one2many, thinning
from Code.reconstruction.one2one_adjust import one2one_candidates
import cv2
import math
import numpy as np


# def adjust_fitting(classify_point, domain_line, match_point, point_line):
def adjust_fitting(path):
    match_point, point_line, background, edge_match_point, edge_point2line, classify_point, domain_line = final(path)

    restoration_line_list = []
    temp_restoration_line_list = []
    for region, point_list in classify_point.items():
        # print(region)
        short_side_point_list = (point_list[0] if (len(point_list[0]) <= len(point_list[1])) else point_list[1])
        if not short_side_point_list:  # 为空
            continue
        domain_line_select(short_side_point_list, domain_line, region, match_point, point_line)
        # print(short_side_point_list)
        ####修补边缘残缺
        for point, matched in edge_match_point.items():
            try:
                # curve fitting of potential bifurcates at the edge of incomplete areas
                segment = one2one(point_line[matched[0]], [point], flag=False)
                segment = thinning(edge_point2line[matched[0]], segment)
            except KeyError:
                # extend fractured ridges at fingerprint edges
                segment = one2one(point_line[point], matched, flag=False)

            restoration_line_list.append(segment)
            domain_line[region].append(segment)
        for i in range(len(short_side_point_list)):
            if i == 0:  ###判断是用domain_line还是其他的
                flag1 = 1
            elif i == len(short_side_point_list) - 1:
                flag1 = 2
            else:
                flag1 = 0
            if short_side_point_list[i] not in match_point.keys():
                continue
            line1 = point_line[short_side_point_list[i]]

            current_restoration_line = []

            if len(match_point[short_side_point_list[i]]) == 2:
                segment1, segment2 = one2many(line1, point_line[match_point[short_side_point_list[i]][0]],
                                              point_line[match_point[short_side_point_list[i]][1]], flag=False)
                current_restoration_line.append(segment1)
                current_restoration_line.append(segment2)
                temp_restoration_line_list.append(
                    line1 + segment1 + point_line[match_point[short_side_point_list[i]][0]])
                temp_restoration_line_list.append(segment2 + point_line[match_point[short_side_point_list[i]][1]])
            else:
                line2 = point_line[match_point[short_side_point_list[i]][0]]
                adjacent_next = []
                if i < len(short_side_point_list) - 1:
                    if len(match_point[short_side_point_list[i + 1]]) == 2:

                        segment1, segment2 = one2many(point_line[short_side_point_list[i + 1]],
                                                      point_line[match_point[short_side_point_list[i + 1]][0]],
                                                      point_line[match_point[short_side_point_list[i + 1]][1]],
                                                      flag=False)
                        adjacent_next.append(segment1)
                        adjacent_next.append(segment2)
                    else:
                        adjacent_next.append(one2one(point_line[short_side_point_list[i + 1]],
                                                     point_line[match_point[short_side_point_list[i + 1]][0]],
                                                     flag=False))

                # range: 控制点计算所用比率取值区间， ratio: 控制点计算比率，数值越大，控制点越远，曲线曲率越大
                # bend: True-->弯曲程度增加, False-->弯曲程度减小, 默认None-->直接用给定比率计算控制点(Default:4)
                # range 和 rate都用返回的那个

                adjacent = find_domain_line(line1[0], line2[0], domain_line, region)

                # 获取若干组候选的修复段
                candidates, info = one2one_candidates(line1, line2, neighbor=adjacent, h=background.shape[0],
                                                      w=background.shape[1])

                distance_list = []
                # print(candidates)
                if candidates:
                    for j in range(len(candidates)):
                        # draw_line(background,candidates[j])
                        aa = candidates[j]
                        dis = judge(candidates[j], domain_line, region, temp_restoration_line_list, flag1,
                                    adjacent_next)
                        distance_list.append(dis)
                        # if dis < float("inf"):
                        #     draw_line(background, candidates[j])

                    # 选择距离最小的
                    if min(distance_list) < float("inf"):
                        current_restoration_line.append(candidates[distance_list.index(min(distance_list))])
                    else:
                        # 不合适的话，选择默认的方式连接
                        current_restoration_line.append(one2one(line1, line2, flag=False))
                else:
                    # 不合适的话，选择默认的方式连接
                    current_restoration_line.append(one2one(line1, line2, flag=False))

                temp_restoration_line_list.append(line1 + current_restoration_line[0] + line2)

            for x in range(len(current_restoration_line)):
                restoration_line_list.append(current_restoration_line[x])
                # domain_line[region].append(current_restoration_line[x])

    return restoration_line_list, background


def draw_line(background, line):
    for po in line:
        background[po[0], po[1]] = 100
    cv2.imshow("ba", background)
    cv2.waitKey(0)


def domain_line_select(short_side_point_list, domain_line, region, match_point, point_line):
    first_line1, last_line1 = point_line[short_side_point_list[0]], point_line[
        match_point[short_side_point_list[0]][0]],
    first_line2, last_line2 = point_line[short_side_point_list[-1]], point_line[
        match_point[short_side_point_list[-1]][0]]
    restoration_line1 = one2one(curve1=first_line1, curve2=last_line1, flag=False)
    restoration_line2 = one2one(curve1=first_line2, curve2=last_line2, flag=False)

    min_distance1, min_distance2 = 10000, 10000
    # for k in range(len(domain_line[region])):
    distance1 = find_min_distance11(restoration_line1, domain_line[region])
    if np.mean(distance1) < min_distance1:
        min_distance1 = np.mean(distance1)
    distance2 = find_min_distance11(restoration_line2, domain_line[region])
    if np.mean(distance2) < min_distance2:
        min_distance2 = np.mean(distance2)
    if min_distance1 > min_distance2:
        short_side_point_list.reverse()


def transfer(line_1):
    for i in range(len(line_1)):
        line_1[i] = (round(line_1[i][0]), round(line_1[i][1]))
    return line_1


def cut_judge(restoration_line, restoration_line_list):
    restoration_line = transfer(restoration_line)
    for x in range(len(restoration_line_list)):
        num = 0
        restoration_line_list[x] = transfer(restoration_line_list[x])
        for z in range(len(restoration_line)):
            if restoration_line[z] in restoration_line_list[x]:
                num += 1
        if num < 0.1 * len(restoration_line) and num > 0:
            return True
    return False


def find_domain_line(point_1, point_2, domain_line, region):
    line_num = 0
    min_distance = 10000
    if not domain_line[region]:
        return None
    else:
        for k in range(len(domain_line[region])):
            distance_1 = []
            distance_2 = []
            line_2 = domain_line[region][k]
            if not line_2:  # 为空跳过
                continue
            for i in range(int(len(line_2))):
                distance_1.append(math.sqrt(
                    (point_1[1] - line_2[i][1]) ** 2 + (point_1[0] - line_2[i][0]) ** 2))
                distance_2.append(math.sqrt(
                    (point_2[1] - line_2[i][1]) ** 2 + (point_2[0] - line_2[i][0]) ** 2))
            distance = [min(distance_1), min(distance_2)]
            if np.mean(distance) < min_distance:
                min_distance = np.mean(distance)
                line_num = k
        return domain_line[region][line_num]


def cal_dis(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def find_min_distance11(line_1, domain_line):
    distance_list = [float("inf")] * 9
    point_list = [line_1[0], line_1[5], line_1[int((len(line_1) / 6))], line_1[2 * int((len(line_1) / 6))],
                  line_1[3 * int((len(line_1) / 6))], line_1[4 * int((len(line_1) / 6))], line_1[
                      5 * int((len(line_1) / 6))], line_1[-5], line_1[-1]]  # 五个点
    for line in domain_line:
        for po in line[::2]:
            for i in range(len(point_list)):
                dis = cal_dis(po, point_list[i])
                if dis <= distance_list[i]:
                    distance_list[i] = dis
    return distance_list


def judge(restoration_line, domain_line, region, restoration_line_list, flag1, adjacent_next):
    if flag1 == 1:  # 第一条
        for line in domain_line[region]:
            intersection = set(restoration_line) & set(line)
            if intersection:
                return float("inf")
        distance_list = find_min_distance11(restoration_line, domain_line[region])
        distance_next_list = find_min_distance11(restoration_line, adjacent_next)
    elif flag1 == 2:  # 末尾
        for line in domain_line[region]:
            intersection = set(restoration_line) & set(line)
            if intersection:
                return float("inf")
        domain_inside_line = restoration_line_list[-2:]
        distance_list = find_min_distance11(restoration_line, domain_line[region])
        distance_next_list = find_min_distance11(restoration_line, domain_inside_line)

    else:  # 中间
        domain_inside_line = restoration_line_list[-2:]
        for line in domain_inside_line:
            intersection = set(restoration_line) & set(line)
            if intersection:
                return float("inf")
        distance_list = find_min_distance11(restoration_line, domain_inside_line)
        distance_next_list = find_min_distance11(restoration_line, adjacent_next)
    if cut_judge(restoration_line, restoration_line_list):
        return float("inf")

    # dis_avg = (distance_list[0]+distance_list[-1])/2
    # dis_sum = abs(distance_list[1]-(dis_avg+distance_list[0])/2) + abs(distance_list[2]-dis_avg) + abs(distance_list[3]-(dis_avg+distance_list[-1])/2)
    dis_sum = 0
    for i in range(2, 7):
        dis_sum += abs(distance_list[i] - (distance_list[-1] - distance_list[0]) / 6 * i - distance_list[0])
        dis_sum += abs(
            distance_next_list[i] - (distance_next_list[-1] - distance_next_list[0]) / 6 * i - distance_next_list[0])
    dis_sum += abs(distance_list[1] - distance_list[0]) + abs(distance_list[-2] - distance_list[-1])
    dis_sum += abs(distance_next_list[1] - distance_next_list[0]) + abs(distance_next_list[-2] - distance_next_list[-1])
    return dis_sum / 7


def thick(pic):
    """
    指纹脊线变粗
    :param path: 细化的图片路径
    :return: 无
    """
    # pic = cv2.imread(path)
    dst = cv2.GaussianBlur(pic, (3, 3), 0)
    # 转换为灰度图像
    # gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    # 转换为二值图像
    ret, binary = cv2.threshold(pic, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 若要细一点可以改为，和下面变细的方法随便选一种就行
    binary = cv2.erode(binary, kernel, iterations=2)  # 若要粗一点，可以增强iterations
    # binary = cv2.dilate(binary, kernel, iterations=1)#若要细一点可以去掉该句注释
    return binary
    # cv2.imshow("11",binary)
    # cv2.waitKey(0)


def draw_whole_img(path):
    # for point in segments:
    #     background[point[0], point[1]] = 150
    segments, background = adjust_fitting(path)
    new_img = background.copy()
    if segments:
        for segment in segments:
            for pos in segment:
                new_img[int(pos[0]), int(pos[1])] = 0

    return background, new_img


def main():
    path_1 = "/home/tuyanglin/fingerprint/reconstruction/train_set/"
    path_2 = "Arch/"
    size = ["large", "small"]
    for i in range(0, 10):
        for si in size:
            if si == "large":
                path = path_1 + path_2 + "Arch_" + str(i + 1) + "_O_v1_" + si + "_noise1" + "_thinned.png"
                print(path)
                background, new_img = draw_whole_img(path)
                cv2.imshow("before restoration", background)
                cv2.imshow("after restoration", new_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                for j in range(3):
                    path = path_1 + path_2 + "Arch_" + str(i + 1) + "_O_v1_" + si + "_noise" + str(
                        j + 1) + "_thinned.png"
                    print(path)
                    background, new_img = draw_whole_img(path)
                    cv2.imshow("before restoration", background)
                    cv2.imshow("after restoration", new_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = '/home/tuyanglin/fingerprint/reconstruction/train_set22/Left/Left_1_O_v1_large_noise1_thinned.png'

    background, new_img = draw_whole_img(path)
    # thick_img = thick(new_img)
    # image_name = os.path.splitext(os.path.basename(path))[0][:-8]

    cv2.imshow("before restoration", background)
    # cv2.imwrite("tututu.png", background)
    cv2.imshow("after restoration", new_img)
    # cv2.imwrite(image_name+"_reconstructed.png",new_img)
    cv2.waitKey(0)

    ###保存为粗的图片
    # path = '/home/tuyanglin/fingerprint/reconstruction/train_set/Arch/Arch_1_O_v1_large_noise1_thinned.png'
    # background, new_img = draw_whole_img(path)
    # thick_img = thick(new_img)
    # image_name = os.path.splitext(os.path.basename(path))[0][:-8]
    # cv2.imwrite(image_name+"_reconstructed.png",thick_img)

    #########脚本 拿细的图片比较
    # path = sys.argv[1]
    # background, new_img = draw_whole_img(path)
    # image_name = os.path.splitext(os.path.basename(path))[0]
    #
    # cv2.imwrite("./tmp/" + image_name + "_pretreat.png", background)
    # cv2.imwrite("./tmp/" + image_name + "_reconstructed.png", new_img)

    #####脚本 拿粗的图片比较,海队
    # path = sys.argv[1]
    # background, new_img = draw_whole_img(path)
    # background = thick(background)
    # new_img = thick(new_img)
    # image_name = os.path.splitext(os.path.basename(path))[0]
    #
    # cv2.imwrite("./incomplete/" + image_name + "_pretreat.png", background)
    # cv2.imwrite("./reconstruct/" + image_name + "_reconstructed.png", new_img)

    #粗的 ,涂杨林
    # path = sys.argv[1]
    # background, new_img = draw_whole_img(path)
    # background = thick(background)
    # new_img = thick(new_img)
    # image_name = os.path.splitext(os.path.basename(path))[0]
    #
    # cv2.imwrite("./tmp/" + image_name + "_pretreat.png", background)
    # cv2.imwrite("./tmp/" + image_name + "_reconstructed.png", new_img)

