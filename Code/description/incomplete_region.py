# encoding: utf-8
import numpy as np
import queue
import cv2
import copy


####残缺区域检测


def delete_stacking_points(img):
    """
    去除细化后图片中堆积点，使之成为单像素宽
    :param img: 输入的numpy
    :return:新的图像
    """
    h, w = img.shape
    template = np.array([[0, 0], [0, 0]])
    for i in np.arange(0, h - 2):
        for j in np.arange(0, w - 2):
            if (img[i:i + 2, j:j + 2] == template).all():
                img[i + 1, j] = 255
                img[i, j + 1] = 255
    return img


def delete_spurious_domain(img, threshold):
    """
    去除噪点连通域
    :param img: 输入的numpy
    :param threshold: 小于threshold的噪点连通域将会被删除
    :return:新的图像
    """
    h, w = img.shape

    # 定义连通域,这里使用8连通
    direction8 = [(-1, 1), (-1, 0), (-1, -1), (0, -1),
                  (1, -1), (1, 0), (1, 1), (0, 1)]

    new_img = img.copy()

    for i in np.arange(1, h - 1):
        for j in np.arange(1, w - 1):  # 扫描整副图像
            if img[i, j] == 0:
                new_flag = False  # 有没有删除噪点的标志
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
                            new_flag = True  #

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


def contours(gray):
    """
    边缘轮廓检测
    输出只带残缺区域的图片
    断点是黑色
    """
    #######对指纹整体轮廓画圈，保证边缘残缺不丢失

    # 转换为二值图像
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    new_binary = copy.deepcopy(binary)  # 外部填充黑色的指纹图
    new_temp = np.ones(binary.shape, dtype=np.uint8) * 255  # 外白内黑的临时图

    # 2次腐蚀,3次膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.erode(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=3)

    cloneImg, contours, heriachy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ###　凸包轮廓检测，保证边缘残缺在里面
    max_contour = contours[0]
    for contour in contours:
        if len(contour) >= len(max_contour):
            max_contour = contour
    hulls = [cv2.convexHull(max_contour)]
    cv2.drawContours(new_temp, hulls, -1, 0, cv2.FILLED)  # 对临时图填充黑色
    # cv2.imshow("1",new_temp)
    new_temp = cv2.dilate(new_temp, kernel, iterations=1)  # erode使得边界的残缺区域识别更大,dilate使得更小,目前都不用.

    ###把指纹外部变为黑色，方便后面的轮廓检测到边缘残缺
    h, w = new_temp.shape
    for i in range(h):
        for j in range(w):
            if new_temp[i][j] == 255:
                new_binary[i][j] = 0
    # cv2.imshow("3",new_binary)

    # 2次腐蚀,3次膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    new_binary = cv2.erode(new_binary, kernel, iterations=3)
    new_binary = cv2.dilate(new_binary, kernel, iterations=3)

    cloneImg, contours, heriachy = cv2.findContours(
        new_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv2.drawContours(gray, contours, i, (0, 0, 255), 1)  # 分别对各轮廓画线
    # cv2.imshow("4",gray)

    new_img = np.ones(binary.shape, dtype=np.uint8) * 255  # 创建白色背景
    # cv2.drawContours(new_img, contours, -1, 0, cv2.FILLED) #对轮廓内部填充黑色
    for i, contour in enumerate(contours):
        cv2.drawContours(new_img, contours, i, i, cv2.FILLED)  # 分别对各轮廓画线
    # cv2.imshow("5",new_img)

    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_img = cv2.erode(new_img, new_kernel, iterations=3)  # 对白色背景腐蚀，使黑色扩大，保证断点在黑色里面
    # cv2.imshow("6", new_img)
    # cv2.waitKey(0)
    return new_img


def judge_derivative(point, first_point, last_point):  # 判断一个点是不是转折点
    der_1 = (point[0] - first_point[0]) / (point[1] - first_point[1] + 0.1)
    der_2 = (point[0] - last_point[0]) / (point[1] - last_point[1] + 0.1)
    if abs(der_1) > 2 or abs(der_2) > 2:
        der_1 = (point[1] - first_point[1]) / (point[0] - first_point[0] + 0.1)
        der_2 = (point[1] - last_point[1]) / (point[0] - last_point[0] + 0.1)
    if abs(der_1 - der_2) > 0.3:  # 阈值
        return True  # 转折点
    else:
        return False  # 平滑点


def judge_del_length(line):
    # 确定在最附近有没有不平滑点
    del_length = 2
    if len(line) > 8:
        for i in range(2, 5):
            if judge_derivative(line[i], line[i - 2], line[i + 2]):
                del_length = i + 1
    return del_length


def remove_breakpoint_bending(img, classify_point, point_line):
    """
    对最后弯曲的点进行去除
    """
    new_classify_point = classify_point
    new_point_line = {}
    for key in classify_point:
        for i, side in enumerate(classify_point[key]):
            for j, po in enumerate(side):
                # del_length = judge_del_length(point_line[po])#没用到,目前是暴力删点法
                del_length = 3
                if len(side) > 2 and len(point_line[po]) > 7:
                    if j == 0 or j == len(side) - 1:
                        del_length = 6
                    if j == 1 or j == len(side) - 2:
                        del_length = 4
                if len(point_line[po])<5:
                    del_length = 0
                new_point_line[point_line[po][del_length]] = point_line[po][del_length:]
                new_classify_point[key][i][j] = point_line[po][del_length]
                for del_po in point_line[po][:del_length]:
                    img[del_po[0], del_po[1]] = 255
    return img, new_classify_point, new_point_line


def connectedDomain(img):
    '''
    img: 输入的numpy
    threshold: 小于threshold的连通域将会被删除
    返回值：图片高宽(h,w),连通域数组
    '''
    # 图片高宽
    h, w = img.shape
    # 定义连通域,这里使用8连通，代表一个像素的8个方向
    direction8 = [(-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]
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
                    connectedDomainGroup[label_number].append((i, j))
                    label[i, j] = 0
                    i_m = i
                    j_m = j
                    branch_flag = True
                    while branch_flag:
                        flag = False
                        for pair in direction8:
                            if label[i_m + pair[0], j_m + pair[1]] == label_number:
                                i_m, j_m = i_m + pair[0], j_m + pair[1]
                                connectedDomainGroup[label_number].append((i_m, j_m))
                                label[i_m, j_m] = 0
                                flag = True
                                break
                        if not flag:
                            branch_flag = False
                elif isBranch == 0:
                    connectedDomainGroup[label[i,j]].append((i,j))
                    label[i,j] = 0


    return connectedDomainGroup, new_label


def cal_distance(point1, point2):  # 计算两点的距离
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def cal_all_distance(point_list):  # 计算点列表所有的距离
    dis = 0
    for i, po in enumerate(point_list):
        if i != len(point_list) - 1:
            dis += cal_distance(po, point_list[i + 1])
    return dis


def insert_point(point, point_list):  ##确定点插入的位置
    min_dis = float("inf")
    length = len(point_list)
    for i in range(length + 1):
        temp_list = point_list.copy()
        temp_list.insert(i, point)
        dis = cal_all_distance(temp_list)
        if dis <= min_dis:
            min_dis = dis
            insert_loc = i
    return insert_loc


def sort_breakpoint(img, classify_point):
    # 对断点进行进行排序
    processed_img = img.copy()
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    new_binary = copy.deepcopy(binary)  # 外部填充黑色的指纹图
    new_temp = np.ones(binary.shape, dtype=np.uint8) * 255  # 外白内黑的临时图

    # 2次腐蚀,3次膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.erode(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=3)

    cloneImg, contours, heriachy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ###　凸包轮廓检测，保证边缘残缺在里面
    max_contour = contours[0]
    for contour in contours:
        if len(contour) >= len(max_contour):
            max_contour = contour
    hulls = [cv2.convexHull(max_contour)]
    cv2.drawContours(new_temp, hulls, -1, 0, cv2.FILLED)  # 对临时图填充黑色

    ###把指纹外部变为黑色，方便后面的轮廓检测到边缘残缺
    h, w = new_temp.shape
    for i in range(h):
        for j in range(w):
            if new_temp[i][j] == 255:
                new_binary[i][j] = 0

    # 2次腐蚀,3次膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    new_binary = cv2.erode(new_binary, kernel, iterations=2)
    new_binary = cv2.dilate(new_binary, kernel, iterations=2)

    cloneImg, contours, heriachy = cv2.findContours(
        new_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    direction8 = [(-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]
    temp_contour = {}  # 轮廓上的点，这个轮廓的区域可能和之前的数量不一样
    temp_point = {}  # 轮廓边缘的断点，按轮廓的顺序存储，把左右两边汇在一起
    new_classify_point = {}
    for i, contour in enumerate(contours):  # 把轮廓进过的点按顺序储存
        # cv2.drawContours(processed_img, contours, i, 0, 1)  # 分别对各轮廓画线
        contour = np.append(contour, [contour[0]], axis=0)  # 把第一个元素加入到最后面，因为是np类型数组，单纯加入不改变，返回值改变．
        temp_contour[i] = []
        for j, po in enumerate(contour):  ###对轮廓上的点提取
            if j != len(contour) - 1:
                [i_m, j_m] = contour[j][0]
                [i_n, j_n] = contour[j + 1][0]
                if i_m != i_n and j_m == j_n:
                    signal = 1 if i_n > i_m else -1
                    X = [i for i in range(i_m, i_n, signal)]
                    for x in X:
                        temp_contour[i].append((j_m, x))
                elif i_m == i_n and j_m != j_n:
                    signal = 1 if contour[j + 1][0][1] > j_m else -1
                    X = [i for i in range(j_m, contour[j + 1][0][1], signal)]
                    for x in X:
                        temp_contour[i].append((x, i_m))
                elif abs(i_m - i_n) == 2 and abs(j_m - j_n) == 2:  # 中间跨一个点
                    temp_contour[i].append((j_m, i_m))
                    temp_contour[i].append(((j_m + j_n) // 2, (i_m + i_n) // 2))

                else:
                    temp_contour[i].append((j_m, i_m))

    ##可以遍历分类点，然后给图片的断点换其他像素．然后对轮廓遍历，把其八联阈的断点按顺序加加来，然后再早出分界点．
    for key in classify_point:  # 标识断点，用不同像素
        temp_point[key] = []
        for num in classify_point[key]:
            for po in num:
                processed_img[po[0], po[1]] = key + 1
    for key in temp_contour:  # 把轮廓旁边的断点存起来
        for po in temp_contour[key]:
            for pair in direction8:
                if processed_img[po[0] + pair[0], po[1] + pair[1]] > 0 and processed_img[
                    po[0] + pair[0], po[1] + pair[1]] != 255:
                    if (po[0] + pair[0], po[1] + pair[1]) not in temp_point[
                        processed_img[po[0] + pair[0], po[1] + pair[1]] - 1]:  # 为断点
                        temp_point[processed_img[po[0] + pair[0], po[1] + pair[1]] - 1].append(
                            (po[0] + pair[0], po[1] + pair[1]))  # 把轮廓旁边的断点存起来
                        break

    # 对temp_contour中的分类,找到一个切分点,然后重新排列，使其为1100,（相当于把1001为0011）
    for key in temp_point:
        count = 0
        while len(temp_point[key]) > 1 and count != len(temp_point[key]) - 1:
            if temp_point[key][count] in classify_point[key][0] and temp_point[key][count + 1] in classify_point[key][
                1] or temp_point[key][count] in classify_point[key][0] and temp_point[key][count + 1] in \
                    classify_point[key][1]:
                temp_point[key] = temp_point[key][count + 1:] + temp_point[key][:count + 1]
                break
            count += 1
    # 对点进行分类，再把其中一类反转
    for key in temp_point:
        new_classify_point[key] = []
        new_classify_point[key].append([])
        new_classify_point[key].append([])
        for po in temp_point[key]:
            if po in classify_point[key][0]:
                new_classify_point[key][0].append(po)
            else:
                new_classify_point[key][1].append(po)
        new_classify_point[key][1].reverse()

    for key in classify_point:  # 增加漏掉的点
        for j, num in enumerate(classify_point[key]):
            for po in num:
                if po not in new_classify_point[key][j] and len(new_classify_point[key][j]) < 2:
                    new_classify_point[key][j].append(po)

    for key in classify_point:  # 增加漏掉的点
        for j, num in enumerate(classify_point[key]):
            for po in num:
                if po not in new_classify_point[key][j]:
                    loc = insert_point(po, new_classify_point[key][j])
                    if loc != 0 and loc != len(new_classify_point[key][j]):
                        new_classify_point[key][j].insert(loc, po)
    return new_classify_point
    # cv2.imshow("22", processed_img)


def judge_line(end_point, key, point_line):  # 应该用比例去判断是什么类型
    """
    判断线是x类型还是y类型，用于分类
    """
    x_count = 0
    y_count = 0
    count = [0, 0, 0, 0]
    for po in end_point[key]:
        length = 4
        if len(point_line[po]) > 4:
            flag = abs(po[0] - point_line[po][length][0]) > abs(po[1] - point_line[po][length][1]) + 1
        else:  # 线太短
            length = len(point_line[po]) - 1
            flag = abs(po[0] - point_line[po][length][0]) > abs(po[1] - point_line[po][length][1])
        if flag:
            y_count += 1
        else:
            x_count += 1

        if po[0] - point_line[po][length][0] > 0:
            if po[1] - point_line[po][length][1] >= 0:
                count[1] += 1
            else:
                count[0] += 1
        else:
            if po[1] - point_line[po][length][1] >= 0:
                count[2] += 1
            else:
                count[3] += 1
    new_count = sorted(enumerate(count), key=lambda x: x[1], reverse=True)
    if x_count / (x_count + y_count) > 0.6:
        return 1  # ｘ类型
    elif x_count / (x_count + y_count) < 0.4:
        return 2  # ｙ类型　数值
    else:
        if abs(new_count[0][0] - new_count[1][0]) != 2:
            return 3  # xy类型
        else:
            if y_count > x_count:
                return 2  # y类型
            else:
                return 1  # x类型


def find_endpoint(img):
    """
    输入图片
    对指纹内部断点进行圈起来
    """
    connectedDomainGroup, new_label = connectedDomain(img)  # 返回的连通域和对应的label
    ###删除较短的线
    for key in connectedDomainGroup:
        if len(connectedDomainGroup[key]) < 3:
            for po in connectedDomainGroup[key]:
                img[po[0], po[1]] = 255
                new_label[po[0], po[1]] = 0

    domain_line = {}  # 储存每个区域的旁边的平行线
    domain_label = {}  # 储存断点那条线的label
    label_pinxing = {}  # 储存残缺区域中其他线的label

    temp_img = img.copy()
    new_img = contours(temp_img)
    # cv2.imshow("1",new_img)
    # cv2.waitKey(0)
    end_point = {}  # 分区域储存断点
    h, w = img.shape
    direction8 = [(-1, 1), (-1, 0), (-1, -1), (0, -1),
                  (1, -1), (1, 0), (1, 1), (0, 1)]
    for i in np.arange(1, h - 1):
        for j in np.arange(1, w - 1):
            # 扫描到未分配连通域的指纹纹路时
            if new_img[i, j] != 255:
                if img[i, j] == 0:
                    isBranch = 0
                    # 计算分叉点
                    for pair in direction8:
                        if img[i + pair[0], j + pair[1]] == 0:
                            isBranch += 1
                    # 判断断点是不是在内部
                    if isBranch == 1:
                        if new_img[i, j] not in end_point.keys():
                            domain_line[new_img[i, j]] = []  # 储存每个区域的平行线
                            end_point[new_img[i, j]] = []
                            end_point[new_img[i, j]].append((i, j))
                            domain_label[new_img[i, j]] = []  # 存储每个区域断点所在的线label
                            domain_label[new_img[i, j]].append(new_label[i, j])
                            label_pinxing[new_img[i, j]] = []
                        else:
                            end_point[new_img[i, j]].append((i, j))
                            domain_label[new_img[i, j]].append(new_label[i, j])
                else:
                    if new_img[i, j] not in domain_line.keys():
                        domain_label[new_img[i, j]] = []
                        label_pinxing[new_img[i, j]] = []
                        domain_line[new_img[i, j]] = []
    # print(new_label[210,230])
    # 判断残缺区域除断点所在的线其他线
    for i in np.arange(1, h - 1):
        for j in np.arange(1, w - 1):
            if new_img[i, j] != 255 and new_label[i, j] != 0 and new_label[i, j] not in domain_label[new_img[i, j]]:
                if new_label[i, j] not in label_pinxing[new_img[i, j]]:
                    label_pinxing[new_img[i, j]].append(new_label[i, j])
                    domain_line[new_img[i, j]].append(connectedDomainGroup[new_label[i, j]])
                    ###画出区域其他线
                    # for pp in connectedDomainGroup[new_label[i,j]]:
                    #     cv2.circle(img, (pp[1], pp[0]), 2, 0, 1)

    # 寻找断点所在的线 !!遇到分叉不会停止
    # print(connectedDomainGroup[new_label[264,66]])
    point_line = {}  # 储存断点所在的线,key是断点坐标
    for value in end_point.values():
        for breakpoint in value:
            if connectedDomainGroup[new_label[breakpoint[0], breakpoint[1]]][0] != breakpoint:
                line_a = connectedDomainGroup[new_label[breakpoint[0], breakpoint[1]]].copy()
                line_a.reverse()
                point_line[breakpoint] = line_a
            else:
                point_line[breakpoint] = connectedDomainGroup[new_label[breakpoint[0], breakpoint[1]]]

    ##对每个残缺区域中一条线有两个断点的进行删除
    for key in end_point:
        for po in end_point[key]:
            if point_line[po][-1] in end_point[key] and len(point_line[po]) < 15:  # 另一头也是断点
                for po1 in point_line[po]:
                    img[po1[0], po1[1]] = 255
                end_point[key].remove(po)
                end_point[key].remove(point_line[po][-1])

    # 对每个区域的点进行分两边
    classify_point = {}  # 分左右储存断点
    for key in end_point:
        if len(end_point[key]) == 0:
            continue
        else:
            classify_point[key] = [[], []]
            if judge_line(end_point, key, point_line) == 1:
                # x类型
                for breakpoint in end_point[key]:
                    length = 4
                    if len(point_line[breakpoint]) <= length:
                        length = len(point_line[breakpoint]) - 1
                    if point_line[breakpoint][0][1] > point_line[breakpoint][length][1]:
                        classify_point[key][0].append(breakpoint)
                        # cv2.circle(img, (breakpoint[1], breakpoint[0]), 2, 0, 1)
                        # temp_img[breakpoint[0]][breakpoint[1]] = [255,255,0]
                    elif point_line[breakpoint][0][1] == point_line[breakpoint][length][1]:
                        if point_line[breakpoint][0][1] > point_line[breakpoint][6][1]:
                            classify_point[key][0].append(breakpoint)
                        else:
                            classify_point[key][1].append(breakpoint)

                    else:
                        classify_point[key][1].append(breakpoint)
                        # cv2.circle(img, (breakpoint[1], breakpoint[0]), 2, 0, 3)
                        # temp_img[breakpoint[0]][breakpoint[1]] = [255, 0, 0]
                classify_point[key][0].sort(key=lambda x: x[0])  # 使点按顺序排列
                classify_point[key][1].sort(key=lambda x: x[0])
            elif judge_line(end_point, key, point_line) == 2:
                # y类型
                for breakpoint in end_point[key]:
                    length = 4
                    if len(point_line[breakpoint]) <= length:
                        length = len(point_line[breakpoint]) - 1
                    if point_line[breakpoint][0][0] > point_line[breakpoint][length][0]:
                        classify_point[key][0].append(breakpoint)
                        # cv2.circle(img, (breakpoint[1], breakpoint[0]), 2, 0, 1)
                        # temp_img[breakpoint[0]][breakpoint[1]] = [255,255,0]
                    else:
                        classify_point[key][1].append(breakpoint)
                        # cv2.circle(img, (breakpoint[1], breakpoint[0]), 2, 0, 2)
                        # temp_img[breakpoint[0]][breakpoint[1]] = [255, 0, 0]
                classify_point[key][0].sort(key=lambda x: x[1])  # 使点按顺序排列
                classify_point[key][1].sort(key=lambda x: x[1])
            else:  # ｘy旋转类型
                temp_classify_point = [[], [], [], []]
                x_count = [0, 0, 0, 0]
                for breakpoint in end_point[key]:
                    length = 4
                    if len(point_line[breakpoint]) <= length:
                        length = len(point_line[breakpoint]) - 1
                    if abs(breakpoint[0] - point_line[breakpoint][length][0]) > abs(
                            breakpoint[1] - point_line[breakpoint][length][1]) + 1:
                        if breakpoint[0] - point_line[breakpoint][length][0] > 0:
                            temp_classify_point[0].append(breakpoint)
                        else:
                            temp_classify_point[1].append(breakpoint)
                    else:  # x类型
                        if breakpoint[1] - point_line[breakpoint][length][1] > 0:
                            temp_classify_point[2].append(breakpoint)
                            if breakpoint[0] - point_line[breakpoint][length][0] > 0:
                                x_count[0] += 1
                            else:
                                x_count[1] += 1
                        else:
                            temp_classify_point[3].append(breakpoint)
                            if breakpoint[0] - point_line[breakpoint][length][0] > 0:
                                x_count[2] += 1
                            else:
                                x_count[3] += 1
                if x_count[0] + x_count[1] > x_count[2] + x_count[3]:
                    if x_count[0] > x_count[1]:
                        classify_point[key][0] = temp_classify_point[0] + temp_classify_point[2]
                        classify_point[key][1] = temp_classify_point[1] + temp_classify_point[3]
                    else:
                        classify_point[key][0] = temp_classify_point[1] + temp_classify_point[2]
                        classify_point[key][1] = temp_classify_point[0] + temp_classify_point[3]
                else:
                    if x_count[2] > x_count[3]:
                        classify_point[key][0] = temp_classify_point[0] + temp_classify_point[3]
                        classify_point[key][1] = temp_classify_point[1] + temp_classify_point[2]
                    else:
                        classify_point[key][0] = temp_classify_point[1] + temp_classify_point[3]
                        classify_point[key][1] = temp_classify_point[0] + temp_classify_point[2]
                # ##画图
                # for po in classify_point[key][0]:
                    # cv2.circle(img, (po[1], po[0]), 2, 0, 1)
                # for po in classify_point[key][1]:
                    # cv2.circle(img, (po[1], po[0]), 2, 0, 2)
    #
    # cv2.imshow("2",img)
    # cv2.waitKey(0)
    classify_point = sort_breakpoint(img, classify_point)
    # 分别为图像，分区域的断点，断点对应的线，每个区域旁边的平行线
    img, classify_point, point_line = remove_breakpoint_bending(img, classify_point, point_line)
    return img, classify_point, point_line, domain_line


def getCoordinate(img):
    """
    path：图片路径
    返回值：图片高宽(h,w),连通域数组
    """
    # img = cv2.imread(path, 0)
    # 二值化
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 去除细化后图片中堆积点，使之成为单像素宽
    img_dsp = delete_stacking_points(img)
    img_dsd = delete_spurious_domain(img_dsp, 3)
    # img_dsw = remove_breakpoint_bending(img_dsd, path)

    # 图像，分区域的断点，断点对应的线，每个区域旁边的平行线
    result, classify_point, point_line, domain_line = find_endpoint(img_dsd)

    # cv2.imshow("Image", result)
    # cv2.imwrite("final.png",result)
    # cv2.waitKey(0)
    return result, classify_point, point_line, domain_line


