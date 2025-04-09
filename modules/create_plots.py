import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import math
import pandas as pd
from .CMC import CMC
import pickle as pkl
from collections import defaultdict

def calculate_d_prime(genuine_scores, imposter_scores):
    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)
    g_mean = np.mean(genuine_scores)
    g_var = np.var(genuine_scores)
    i_mean = np.mean(imposter_scores)
    i_var = np.var(imposter_scores)
    d_prime = abs(g_mean - i_mean)/math.sqrt(0.5 * (g_var + i_var))
    return d_prime

def generate_ytrue_yscore(genuine_scores, imposter_scores):
    y_true = []
    y_score = []
    for score in genuine_scores:
        y_true.append(1)
        y_score.append(1.0 - score)
    for score in imposter_scores:
        y_true.append(0)
        y_score.append(1.0 - score)
    return y_true, y_score

def compute_cmc(search_filenames, enrolled_filenames, scores_dict, topk=30):
    valid_queries = 0
    all_rank = []
    sum_rank = np.zeros(topk)
    for search_filename in tqdm(search_filenames, desc="Finding CMC"):
        # Calculate the distances for each query
        search_uid = search_filename.split('/')[-1].split('+')[0]
        distmat = []
        for enrolled_filename in enrolled_filenames:
            # Get the label from the image
            enrolled_uid = enrolled_filename.split('/')[-1].split('+')[0]
            if (search_filename, enrolled_filename) not in scores_dict:
                continue
            dist = scores_dict[(search_filename, enrolled_filename)]
            distmat.append([dist, enrolled_uid])
        
        distmat.sort()

        # Find matches
        matches = np.zeros(len(distmat))
        # Zero if no match 1 if match
        for i in range(0, len(distmat)):
            if distmat[i][1] == search_uid:
                # Match found
                matches[i] = 1
        rank = np.zeros(topk)
        for i in range(0, topk):
            if matches[i] == 1:
                rank[i] = 1
                # If 1 is found then break as you dont need to look further path k
                break
        all_rank.append(rank)
        valid_queries +=1
    #print(all_rank)
    sum_all_ranks = np.zeros(len(all_rank[0]))
    for i in range(0,len(all_rank)):
        my_array = all_rank[i]
        for g in range(0, len(my_array)):
            sum_all_ranks[g] = sum_all_ranks[g] + my_array[g]
    sum_all_ranks = np.array(sum_all_ranks)
    cmc_results = np.cumsum(sum_all_ranks) / valid_queries
    return cmc_results, sum_all_ranks

def generate_plots(scores_csv="scores.csv", cmc_rank=10, skip_roc=False, skip_cmc=False, roc_out="roc.png", cmc_out="cmc.png"):
    scores_df = pd.read_csv(scores_csv)
    score_names = scores_df.columns[2:].sort_values()

    genuine_scores = defaultdict(list)
    imposter_scores = defaultdict(list)

    search_filenames = []
    enroll_filenames = []
    scores_dict = defaultdict(dict)

    for _, row in tqdm(scores_df.iterrows(), desc="Loading scores", total=len(scores_df)):
        enroll_filename = row["Enroll"]
        search_filename = row["Search"]
        enroll_uid = enroll_filename.split('/')[-1].split('+')[0]
        search_uid = search_filename.split('/')[-1].split('+')[0]
        search_filenames.append(search_filename)
        enroll_filenames.append(enroll_filename)
        for score_name in score_names:
            if enroll_uid == search_uid:
                genuine_scores[score_name].append(float(row[score_name]))
            else:
                imposter_scores[score_name].append(float(row[score_name]))
            scores_dict[score_name][(search_filename, enroll_filename)] = float(row[score_name])

    if not skip_roc:
        plt.figure(figsize=(10, 8 + len(score_names) // 4))
        plt.title("ROC")
        for score_name in score_names:
            d_prime = calculate_d_prime(genuine_scores[score_name], imposter_scores[score_name])
            y_true, y_score = generate_ytrue_yscore(genuine_scores[score_name], imposter_scores[score_name])
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            plt.plot(fpr, tpr, label=f"{score_name} [AUC={round(auc, 4)}] [d'={round(d_prime, 4)}]")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xscale('log')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        plt.savefig(roc_out, bbox_inches='tight')
        plt.close()

    if not skip_cmc:
        if os.path.exists("cmc_results.pkl"):
            with open('cmc_results.pkl', 'rb') as f:
                cmc_results = pkl.load(f)
        else:
            cmc_results = {}
        for score_name in score_names:
            if score_name in cmc_results:
                continue
            cmc_result, _ = compute_cmc(search_filenames, enroll_filenames, scores_dict[score_name], 10)
            cmc_results[score_name] = cmc_result

        with open('cmc_results.pkl', 'wb') as f:
            pkl.dump(cmc_results, f)

        default_colors = ['r','g','b','c','m','y','orange','brown']
        default_markers = ['*','o','s','v','X','*','.','P']
        cmc_result = CMC(cmc_results, color=default_colors[:len(score_names)], marker=default_markers[:len(score_names)])
        filename, format = os.path.splitext(cmc_out)
        cmc_result.save(title = 'CMC Results', rank=cmc_rank,
                xlabel='Rank Score',
                ylabel='Recognition Rate', show_grid=True, 
                filename=filename, format=format.replace('.', ''))
