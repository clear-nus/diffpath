import numpy as np
import argparse 
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
import os

def main():

    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--in_dist', type=str, required=True, help='In distribution')
    parser.add_argument('--out_of_dist', type=str, required=True, help='Out of distribution type')
    parser.add_argument('--n_ddim_steps', type=int, default=50, help='Number of ddim steps')
    args = parser.parse_args()
    
    # load in distribution training dataset statistics
    in_dist_train_path = os.path.join(f"train_statistics/ddim{args.n_ddim_steps}", args.in_dist) + ".npz"
    print(f"Loading in dist train statistics from {in_dist_train_path}")
    in_dist_train_statistics_file = np.load(in_dist_train_path)
    id_train_eps = in_dist_train_statistics_file["eps_sum"]
    id_train_eps_sq = in_dist_train_statistics_file["eps_sum_sq"]
    id_train_eps_cb = in_dist_train_statistics_file["eps_sum_cb"]
    id_train_deps_dt = in_dist_train_statistics_file["deps_dt"]
    id_train_deps_dt_sq = in_dist_train_statistics_file["deps_dt_sq"]
    id_train_deps_dt_cb = in_dist_train_statistics_file["deps_dt_cb"]
    id_train_statistics = np.column_stack([id_train_eps, id_train_eps_sq, id_train_eps_cb,
                                           id_train_deps_dt, id_train_deps_dt_sq, id_train_deps_dt_cb])
    print("Fitting GMM to in dist training set")
    gmm = GaussianMixture(n_components=50, covariance_type='diag').fit(id_train_statistics)
    
    # load in distribution test dataset statistics
    in_dist_test_path = os.path.join(f"test_statistics/ddim{args.n_ddim_steps}", args.in_dist) + ".npz"
    print(f"Loading in dist test statistics from {in_dist_test_path}")
    in_dist_test_statistics_file = np.load(in_dist_test_path)
    id_test_eps = in_dist_test_statistics_file["eps_sum"]
    id_test_eps_sq = in_dist_test_statistics_file["eps_sum_sq"]
    id_test_eps_cb = in_dist_test_statistics_file["eps_sum_cb"]
    id_test_deps_dt = in_dist_test_statistics_file["deps_dt"]
    id_test_deps_dt_sq = in_dist_test_statistics_file["deps_dt_sq"]
    id_test_deps_dt_cb = in_dist_test_statistics_file["deps_dt_cb"]
    id_test_statistics = np.column_stack([id_test_eps, id_test_eps_sq, id_test_eps_cb,
                                          id_test_deps_dt, id_test_deps_dt_sq, id_test_deps_dt_cb])
    score_test_id = gmm.score_samples(id_test_statistics)
    
    # load out of distribution test dataset statistics
    out_of_dist_test_path = os.path.join(f"test_statistics/ddim{args.n_ddim_steps}", args.out_of_dist) + ".npz"
    print(f"Loading out of dist test statistics from {out_of_dist_test_path}")
    ood_test_statistics_file = np.load(out_of_dist_test_path)
    ood_eps = ood_test_statistics_file["eps_sum"]
    ood_eps_sq = ood_test_statistics_file["eps_sum_sq"]
    ood_eps_cb = ood_test_statistics_file["eps_sum_cb"]
    ood_deps_dt = ood_test_statistics_file["deps_dt"]
    ood_deps_dt_sq = ood_test_statistics_file["deps_dt_sq"]
    ood_deps_dt_cb = ood_test_statistics_file["deps_dt_cb"]
    ood_statistics = np.column_stack([ood_eps, ood_eps_sq, ood_eps_cb,
                                      ood_deps_dt, ood_deps_dt_sq, ood_deps_dt_cb])
    score_ood = gmm.score_samples(ood_statistics)
    
    y_test_id = np.ones(score_test_id.shape[0])
    y_test_ood = np.zeros(score_ood.shape[0])
    y_true = np.append(y_test_id, y_test_ood)

    sample_score = np.append(score_test_id, score_ood)
    auroc = roc_auc_score(y_true, sample_score)

    print(f"In dist: {args.in_dist}, Out of dist: {args.out_of_dist}, DiffPath 6D AUROC: {auroc}")

if __name__ == "__main__":
    main()