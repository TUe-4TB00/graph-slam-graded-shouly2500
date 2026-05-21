import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    
    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None     # chosen pose option
    best_landmark = None   # chosen landmark (1 or 2)
    best_select = float("inf") 
    best_report = None

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = graph.clone()
            est = gtsam.Values(initial_estimate)
            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            marginals = gtsam.Marginals(g, result)
            select = marginals.marginalCovariance(L(1)).trace() + marginals.marginalCovariance(L(2)).trace()
            report = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()
            if select < best_select:
                best_select = select
                best_report = report
                best_pose = pose_key
                best_landmark = landmark

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    marginals = gtsam.Marginals(graph, result)
    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    sum_of_marginals = best_report
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None      # chosen pose option
    best_landmark = None   # chosen landmark (1 or 2)
    sum_of_errors = float("inf")
    truth = {i: np.array([2.0 * (i-1), 0.0]) for i in [1,2,3]}
    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = graph.clone()
            est = gtsam.Values(initial_estimate)
            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)
            list_of_errors = []
            for i in [1,2,3]:
                list_of_errors.append(np.linalg.norm(result.atPose2(X(i)).translation() - truth[i]))
            current = sum(list_of_errors)
            if current < sum_of_errors:
                sum_of_errors = current
                best_pose = pose_key
                best_landmark = landmark
     # TODO: compute the sum of the errors and return it along with the best pose and landmark
    return best_pose, best_landmark, sum_of_errors 