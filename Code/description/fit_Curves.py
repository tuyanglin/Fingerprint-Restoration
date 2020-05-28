from numpy import *
import copy

'''
fitCurve:曲线生成控制点
bezier_curve_fit：控制点生成曲线
'''

def q(ctrlPoly, t):
    return (1.0-t)**3 * ctrlPoly[0] + 3*(1.0-t)**2 * t * ctrlPoly[1] + 3*(1.0-t)* t**2 * ctrlPoly[2] + t**3 * ctrlPoly[3]


# evaluates cubic bezier first derivative at t, return point
def qprime(ctrlPoly, t):
    return 3*(1.0-t)**2 * (ctrlPoly[1]-ctrlPoly[0]) + 6*(1.0-t) * t * (ctrlPoly[2]-ctrlPoly[1]) + 3*t**2 * (ctrlPoly[3]-ctrlPoly[2])


# evaluates cubic bezier second derivative at t, return point
def qprimeprime(ctrlPoly, t):
    return 6*(1.0-t) * (ctrlPoly[2]-2*ctrlPoly[1]+ctrlPoly[0]) + 6*(t) * (ctrlPoly[3]-2*ctrlPoly[2]+ctrlPoly[1])


# Fit one (ore more) Bezier curves to a set of points
def fitCurve(points, maxError):
    leftTangent = normalize(points[1] - points[0])
    rightTangent = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTangent, maxError)


def fitCubic(points, leftTangent, rightTangent, error):
    # Use heuristic if region only has two points in it
    if (len(points) == 2):
        dist = linalg.norm(points[0] - points[1]) / 3.0
        bezCurve = [points[0], points[0] + leftTangent * dist, points[1] + rightTangent * dist, points[1]]
        return [bezCurve]

    # Parameterize points, and attempt to fit curve
    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    # Find max deviation of points to fitted curve
    maxError, splitPoint = computeMaxError(points, bezCurve, u)
    if maxError < error:
        return [bezCurve]

    # If error not too large, try some reparameterization and iteration
    if maxError < error**2:
        for i in range(20):
            uPrime = reparameterize(bezCurve, points, u)
            bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
            maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
            if maxError < error:
                return [bezCurve]
            u = uPrime

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    centerTangent = normalize(points[splitPoint-1] - points[splitPoint+1])
    beziers += fitCubic(points[:splitPoint+1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)

    return beziers


def generateBezier(points, parameters, leftTangent, rightTangent):
    bezCurve = [points[0], None, None, points[-1]]

    # compute the A's
    A = zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = leftTangent  * 3*(1-u)**2 * u
        A[i][1] = rightTangent * 3*(1-u)    * u**2

    # Create the C and X matrices
    C = zeros((2, 2))
    X = zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += dot(A[i][0], A[i][0])
        C[0][1] += dot(A[i][0], A[i][1])
        C[1][0] += dot(A[i][0], A[i][1])
        C[1][1] += dot(A[i][1], A[i][1])

        tmp = point - q([points[0], points[0], points[-1], points[-1]], u)

        X[0] += dot(A[i][0], tmp)
        X[1] += dot(A[i][1], tmp)

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X  = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1  = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text) */
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent NewtonRaphsonRootFind() call. */
    segLength = linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * segLength
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bezCurve[1] = bezCurve[0] + leftTangent * (segLength / 3.0)
        bezCurve[2] = bezCurve[3] + rightTangent * (segLength / 3.0)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bezCurve[1] = bezCurve[0] + leftTangent * alpha_l
        bezCurve[2] = bezCurve[3] + rightTangent * alpha_r

    return bezCurve


def reparameterize(bezier, points, parameters):
    return [newtonRaphsonRootFind(bezier, point, u) for point, u in zip(points, parameters)]


def newtonRaphsonRootFind(bez, point, u):
    """
       Newton's root finding algorithm calculates f(x)=0 by reiterating
       x_n+1 = x_n - f(x_n)/f'(x_n)

       We are trying to find curve parameter u for some point p that minimizes
       the distance from that point to the curve. Distance point to curve is d=q(u)-p.
       At minimum distance the point is perpendicular to the curve.
       We are solving
       f = q(u)-p * q'(u) = 0
       with
       f' = q'(u) * q'(u) + q(u)-p * q''(u)

       gives
       u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    """
    d = q(bez, u)-point
    numerator = (d * qprime(bez, u)).sum()
    denominator = (qprime(bez, u)**2 + d * qprimeprime(bez, u)).sum()

    if denominator == 0.0:
        return u
    else:
        return u - numerator/denominator


def chordLengthParameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i-1] + linalg.norm(points[i] - points[i-1]))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]

    return u


def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points)/2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = linalg.norm(q(bez, u)-point)**2
        if dist > maxDist:
            maxDist = dist
            splitPoint = i

    return maxDist, splitPoint


def normalize(v):
    return v / linalg.norm(v)

def dijkstra(pos_list):
    """
    对贝塞尔曲线拟合出来的线进行细化
    :param pos_list:
    :return: 细化后的线
    """
    direction8 = [(-1, 1), (-1, 0), (-1, -1), (0, -1),
                  (1, -1), (1, 0), (1, 1), (0, 1)]
    graph = ones((len(pos_list), len(pos_list))) * inf

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
        len_min = inf
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

def bezier_curve_fit(points, t_num=1000):
    """
    贝塞尔曲线拟合，需要传入四个点
    :param points: 四个点
    :param t_num: 绘制的点的个数，默认为100个，满足大部分图像大小
    :return: 拟合后的曲线
    """
    bx = 3 * (points[1][0] - points[0][0])
    cx = 3 * (points[2][0] - points[1][0]) - bx
    dx = points[3][0] - points[0][0] - bx - cx
    by = 3 * (points[1][1] - points[0][1])
    cy = 3 * (points[2][1] - points[1][1]) - by
    dy = points[3][1] - points[0][1] - by - cy

    t = linspace(0, 1, t_num)

    xx = points[0][0] + bx * t + cx * t ** 2 + dx * t ** 3
    yy = points[0][1] + by * t + cy * t ** 2 + dy * t ** 3

    segment = list(zip(xx, yy))

    segment = [(int(x), int(y)) for (x, y) in segment]
    segment = sorted(set(segment), key=segment.index)

    return dijkstra(segment)
