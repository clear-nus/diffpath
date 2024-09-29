import numpy as np
import argparse 
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
import os


def load_statistics(d):
    s = d["deps_dt_sq_sqrt"]
    return s


def main():

    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--model', type=str, default='celeba', help="Base distribution of model")
    parser.add_argument('--in_dist', type=str, required=True, help='In distribution')
    parser.add_argument('--out_of_dist', type=str, required=True, help='Out of distribution type')
    parser.add_argument('--n_ddim_steps', type=int, default=10, help='Number of ddim steps')
    args = parser.parse_args()
    
    # load in distribution training dataset statistics
    kde_bandwith = 5
    in_dist_train_path = os.path.join(f"train_statistics_{args.model}_model/ddim{args.n_ddim_steps}", args.in_dist) + ".npz"
    print(f"Loading in dist train statistics from {in_dist_train_path}")
    in_dist_train_statistics_file = np.load(in_dist_train_path)
    id_train_deps_dt_sq_sqrt = load_statistics(in_dist_train_statistics_file).reshape(-1,1)
    print("Fitting KDE to in dist training set")
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwith).fit(id_train_deps_dt_sq_sqrt) 
    
    # load in distribution test dataset statistics
    in_dist_test_path = os.path.join(f"test_statistics_{args.model}_model/ddim{args.n_ddim_steps}", args.in_dist) + ".npz"
    print(f"Loading in dist test statistics from {in_dist_test_path}")
    in_dist_test_statistics_file = np.load(in_dist_test_path)
    id_test_deps_dt_sq_sqrt = load_statistics(in_dist_test_statistics_file).reshape(-1,1)
    score_test_id = kde.score_samples(id_test_deps_dt_sq_sqrt)
    
    # load out of distribution test dataset statistics
    out_of_dist_test_path = os.path.join(f"test_statistics_{args.model}_model/ddim{args.n_ddim_steps}", args.out_of_dist) + ".npz"
    print(f"Loading out of dist test statistics from {out_of_dist_test_path}")
    ood_test_statistics_file = np.load(out_of_dist_test_path)
    ood_test_deps_dt_sq_sqrt = load_statistics(ood_test_statistics_file).reshape(-1,1)
    score_ood = kde.score_samples(ood_test_deps_dt_sq_sqrt)
    
    y_test_id = np.ones(score_test_id.shape[0])
    y_test_ood = np.zeros(score_ood.shape[0])
    y_true = np.append(y_test_id, y_test_ood)

    sample_score = np.append(score_test_id, score_ood)
    auroc = roc_auc_score(y_true, sample_score)

    print(f"In dist: {args.in_dist}, Out of dist: {args.out_of_dist}, DiffPath 1D AUROC: {auroc}")

if __name__ == "__main__":
    main()