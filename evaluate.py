import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):

        if (actual[max(i, 0) : i + 1]).any() and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j].any():
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i].any():
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
    

def calc_point2point_point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    actual = actual.reshape(-1)
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN, predict


def calc_seq(score, label, threshold):
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point_point(predict.astype(int), label), latency


def bf_search_point(score, label, start, end, step_num):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    if step_num is None or end is None:
        end = start
        step_num = 1

    search_step, search_range, search_lower_bound = step_num, end - start, start
    threshold = search_lower_bound

    m = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0

    for i in tqdm(range(search_step), total=search_step, desc="Finding the best threshold"):
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m = target
            m_t = threshold
            m_l = latency

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "threshold": m_t,
        "latency": m_l
    }


def evaluation(args, test_results):
    T, N = test_results[0].shape
    label = test_results[-2]
    label_ts = (np.sum(label, axis=1) >= 1) + 0
    att_score = test_results[-1]

    batch = []
    for t in range(T):
        if t % 256 == 0:
            if t-5 > 0:
                batch.append(t-5)
            if t-4 > 0:
                batch.append(t-4)
            if t-3 > 0:
                batch.append(t-3)
            if t-2 > 0:
                batch.append(t-2)
            if t-1 > 0:
                batch.append(t-1)
            batch.append(t)
            if t+1 < T:
                batch.append(t+1)
            if t+2 < T:
                batch.append(t+2)
            if t+3 < T: 
                batch.append(t+3)
            if t+4 < T:
                batch.append(t+4)
            if t+5 < T:
                batch.append(t+5)

    anom_start_t = []
    anom_end_t = []

    for t in range(len(label_ts)-1):
        if label_ts[t] == 0 and label_ts[t+1] == 1:
            anom_start_t.append(t+1)
        elif label_ts[t] == 1 and label_ts[t+1] == 0:
            anom_end_t.append(t+1)
        else:
            continue

    if len(anom_start_t) > len(anom_end_t):
        anom_start_t = anom_start_t[:len(anom_end_t)]
    elif len(anom_end_t) > len(anom_start_t):
        anom_end_t = anom_end_t[:len(anom_start_t)]

    anom_len = []
    anom_cause = []
    for t in range(len(anom_start_t)):
        if anom_end_t[t] - anom_start_t[t] < 0:
            print("error")
            break
        anom_len.append(anom_end_t[t] - anom_start_t[t])
        anom_cause.append(np.where(label[anom_start_t[t]] == 1)[0].tolist())

    anom_seg = {(anom_start_t[i], anom_end_t[i]): (anom_len[i], anom_cause[i]) for i in range(len(anom_start_t))}

    def js_divergence(p, q, epsilon=1e-12):
        p = p.clamp(min=epsilon)
        q = q.clamp(min=epsilon)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        m += epsilon
        p += epsilon
        q += epsilon
        
        kl_pm = F.kl_div(m.log(), p, reduction='batchmean', log_target=False)
        kl_qm = F.kl_div(m.log(), q, reduction='batchmean', log_target=False)
        
        return 0.5 * (kl_pm + kl_qm)

    scores = torch.zeros(T, N)

    for node_idx in tqdm(range(N)):
        node_att_score = torch.tensor(att_score[:,node_idx,:])
        js_distances = torch.zeros(T)
        for t in range(T-1):
            js_distances[t+1] = js_divergence(node_att_score[t], node_att_score[t+1])
        scores[:,node_idx] = js_distances

    scores = np.array(scores)
    temp_scores = scores.copy()
    temp_scores[batch,:] = 0

    temp_scores_v2 = np.zeros_like(temp_scores)

    for node_idx in tqdm(range(N)):
        time_series = temp_scores[:, node_idx]
        fft_result = np.fft.rfft(time_series)
        frequencies = np.fft.rfftfreq(len(time_series), d=0.1)
        dominant_frequency = frequencies[np.argmax(np.abs(fft_result)[1:]) + 1]
        dominant_period = int(round(1 / dominant_frequency))
        if dominant_period < 2:
            dominant_period = int(2 * np.pi / 0.1)
        # period = int(2 * np.pi / 0.1)
        
        stl = STL(time_series, period=dominant_period, seasonal=13)
        result = stl.fit()

        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        deseasonalized = time_series - seasonal

        temp_scores_v2[:, node_idx] = deseasonalized

    max_scores = torch.max(torch.tensor(temp_scores_v2), dim=-1).values
    max_scores = np.array(max_scores)
    ad_results_max = bf_search_point(max_scores, 
                                    label_ts, 
                                    start=np.quantile(max_scores,0.8), 
                                    end=np.quantile(max_scores,1), 
                                    step_num=100)

    hr100_list = []
    hr150_list = []

    for (sa, ea), (la, causes) in list(anom_seg.items()):
        s = temp_scores_v2[sa:ea, :]
        max_node = np.argmax(np.max(s, axis=0))

        target_series = s[:, max_node]
        correlations = []

        if len(target_series) == 1:
            for i in range(s.shape[0]):
                a = s[i]
                a = np.argsort(a).tolist()[::-1]
                size100 = round(100 * len(causes) / 100)
                size150 = round(150 * len(causes) / 100)
                a_100 = set(a[:size100])
                a_150 = set(a[:size150])
                intersect100 = a_100.intersection(set(causes))
                intersect150 = a_150.intersection(set(causes))
                hit100 = len(intersect100) / len(causes)
                hit150 = len(intersect150) / len(causes)
                hr100_list.append(hit100)
                hr150_list.append(hit150)

        else:
            for i in range(N):
                correlation = np.corrcoef(target_series, s[:, i])[0, 1]
                correlations.append((i, correlation))

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            correlations_index = []
            for index, correlation in correlations:
                correlations_index.append(index)

            size100 = round(100 * len(causes) / 100)
            size150 = round(150 * len(causes) / 100)
            a_100 = set(correlations_index[:size100])
            a_150 = set(correlations_index[:size150])
            intersect100 = a_100.intersection(set(causes))
            intersect150 = a_150.intersection(set(causes))
            hit100 = len(intersect100) / len(causes)
            hit150 = len(intersect150) / len(causes)

            for _ in range(la):
                hr100_list.append(hit100)
                hr150_list.append(hit150)

    res = {}
    res['Hit@100%'] = np.mean(hr100_list)
    res['Hit@150%'] = np.mean(hr150_list)
    for k, v in res.items():
        res[k] = float(v)

    return (ad_results_max, res, (temp_scores_v2, label))