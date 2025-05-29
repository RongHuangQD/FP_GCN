from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score

from HiGCN import HiGCN
from data_loading import load_graph_dataset, DataLoader
from parser import get_parser
import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import random
import time
from train_utils import train, eval
import seaborn as sns
import matplotlib.pyplot as plt
project_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(project_dir, '../data', 'poi.data')
def main(args):
    """The common training and evaluation script used by all the experiments."""
    # set device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    print("==========================================================")
    print("Using device", str(device))
    print(f"Seed: {args.seed}")
    print("======================== Args ===========================")
    print(args)
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load graph dataset
    graph_list, train_ids, test_ids, num_classes = load_graph_dataset(
        args.dataset, max_petal_dim=args.max_petal_dim)

    # 将代码嵌入进去，HL保存到graph_list中
    train_graphs = [graph_list[i] for i in train_ids]
    test_graphs = [graph_list[i] for i in test_ids]
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    num_features = graph_list[0].x.shape[1]
    # Instantiate model
    model = HiGCN(args.max_petal_dim,
                num_features,
                args.num_layers,
                args.hidden,
                num_classes,
                dropout_rate=args.drop_rate,
    ).to(device)

    print("============= Model Parameters =================")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            trainable_params += param.numel()
        total_params += param.numel()
    print("============= Params stats ==================")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params    : {total_params}")

    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # instantiate learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_scheduler_decay_steps,gamma=args.lr_scheduler_decay_rate)

    # (!) start training/evaluation
    train_curve = []
    train_loss_curve = []
    params = []
    epoch_times = []
    if not args.untrained:
        for epoch in range(1, args.epochs + 1):
            # perform one epoch
            print("=====Epoch {}".format(epoch))
            print('Training...')
            start_time = time.time()
            epoch_train_curve = train(model, device, train_loader, optimizer)
            train_loss_curve += epoch_train_curve
            epoch_train_loss = float(np.mean(epoch_train_curve))
            # # evaluate model
            #print('Evaluating...')
            train_loader_eval = DataLoader(train_graphs, batch_size=args.batch_size,shuffle=False)
            test_loader_eval = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
            train_perf, _, train_true, train_pred = eval(model, device, train_loader_eval)
            train_curve.append(train_perf)
            end_time = time.time()
            epoch_times.append(end_time - start_time)

            print(f'Train: {train_perf:.3f} '
                  f' | Train Loss {epoch_train_loss:.3f} ')

            # decay learning rate
            scheduler.step()
            i = 0
            new_params = []
            if epoch % args.train_eval_period == 0:
                print("====== Slowly changing params ======= ")
            for name, param in model.named_parameters():
                new_params.append(param.data.detach().mean().item())
                if len(params) > 0 and epoch % args.train_eval_period == 0:
                    if abs(params[i] - new_params[i]) < 1e-6:
                        print(f"Param {name}: {params[i] - new_params[i]}")
                i += 1
            params = copy.copy(new_params)
        confusion_mat_train = confusion_matrix(train_true, train_pred)
        print("Confusion matrix for train dataset:")
        print(confusion_mat_train)
        train_class_samples = np.sum(confusion_mat_train, axis=1)

        normalized_confusion_mat_train = confusion_mat_train / train_class_samples[:, np.newaxis]
        print("Confusion matrix for train dataset containing precision:")
        for row in normalized_confusion_mat_train:
            print(" ".join(["{:.3f}".format(val) for val in row]))
    else:
        train_loss_curve.append(np.nan)#"not a number"
        train_curve.append(np.nan)
    print('Evaluation...')
    # final_train_perf = np.nan
    # test_perf = np.nan
    test_perf, _, test_true, test_pred = eval(model, device, test_loader_eval)
    # Calculate confusion matrix for the final evaluation
    confusion_mat_test = confusion_matrix(test_true, test_pred)
    print("Confusion matrix for test dataset:")
    print(confusion_mat_test)
    test_class_samples = np.sum(confusion_mat_test, axis=1)
    normalized_confusion_mat_test = confusion_mat_test / test_class_samples[:, np.newaxis]
    print("Confusion matrix for test dataset containing precision:")
    for row in normalized_confusion_mat_test:
        print(" ".join(["{:.3f}".format(val) for val in row]))
    class_report_train = classification_report(train_true, train_pred)
    class_report_test = classification_report(test_true, test_pred)
    print("class_report:")
    print(class_report_train)
    print(class_report_test)
    weighted_train_matrix = normalized_confusion_mat_train * 0.7
    weighted_test_matrix = normalized_confusion_mat_test * 0.3

    combined_matrix = weighted_train_matrix + weighted_test_matrix
    print("Confusion matrix for all dataset containing precision:\n")
    matrix_str = ""
    for row1 in combined_matrix:
        row_str = " ".join(["{:.3f}".format(val) for val in row1])
        matrix_str += row_str + "\n"
    print(matrix_str)

    combined_true = np.concatenate((train_true, test_true))
    combined_pred = np.concatenate((train_pred, test_pred))
    kappa_combined = cohen_kappa_score(combined_true, combined_pred)
    print(f"Kappa: {kappa_combined:.4f}")

    epoch_times = np.array(epoch_times)
    # save results
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'last_test': test_perf,
        'last_train': train_perf,
        'epoch_time': np.mean(epoch_times) }

    msg = (
       f'========== Result ============\n'
       f'epoch_time:     {np.mean(epoch_times)} ± {np.std(epoch_times)}\n\n'
       '------------ Last epoch -----------\n'
       f'Train:          {train_perf}\n'
       f'Test:           {test_perf}\n'
       f'Average:        {train_perf*0.7+test_perf*0.3}\n'
       f'Confusion matrix for all dataset containing precision:\n {matrix_str}\n'
       '-------------------------------\n')
    print(msg)

    msg += str(args)

    # Create results folder
    result_folder = os.path.join(args.result_folder, f'{args.dataset}-{args.exp_name}')
    result_folder = os.path.join(result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'result.txt')

    with open(filename, 'w') as handle:
        handle.write(msg)
    if args.dump_curves:
        with open(os.path.join(result_folder, 'curves.pkl'), 'wb') as handle:
            pickle.dump(curves, handle)

    # generate true and predicted label file
    output_file = os.path.join(result_folder, 'true_pred.txt')
    with open(output_file, 'w') as file:
        for zone_id, true_label, pred_label in zip(train_ids, train_true, train_pred):
            file.write(f"Zone ID: {zone_id}, True Label: {true_label}, Predicted Label: {pred_label}\n")
        for zone_id, true_label, pred_label in zip(test_ids, test_true, test_pred):
            file.write(f"Zone ID: {zone_id}, True Label: {true_label}, Predicted Label: {pred_label}\n")
    with open(output_file, 'r') as file:
        lines = file.readlines()
    sorted_lines = sorted(lines, key=lambda line: int(line.split(',')[0].split(': ')[1]))
    sorted_file = os.path.join(result_folder, 'sorted_true_pred.txt')
    with open(sorted_file, 'w') as file:
        file.writelines(sorted_lines)

    #Confusion matrix
    matrix_str = matrix_str.splitlines()
    combined_matrix = np.array([list(map(float, row.split())) for row in matrix_str])
    plt.rcParams['font.family'] = 'Times New Roman ,SimSun '  # 设置字体族，中文为SimSun，英文为Times New Roman
    classes = ['V', 'VLE', 'R', 'RC', 'BH', 'I', 'C', 'LE', 'RCE', 'A', 'F']
    new_order = ['A', 'BH', 'C', 'F', 'I', 'LE', 'R', 'RC', 'RCE', 'V', 'VLE']
    row_indices = [classes.index(cls) for cls in new_order]
    reordered_matrix = combined_matrix[row_indices][:, row_indices]
    plt.figure(figsize=(10, 8))

    #heatmap
    heatmap = sns.heatmap(reordered_matrix, annot=True, fmt=".3f", cmap="OrRd", xticklabels=new_order,
                          yticklabels=new_order, annot_kws={'size': 14}, cbar=True,
                          square=True)  # linewidths=.5,linecolor='black'
    heatmap.set_yticklabels(heatmap.get_yticklabels(), weight='bold', rotation=0)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), weight='bold')
    # add title and label
    plt.title("Overall Confusion Matrix", fontsize=18, pad=10)
    plt.xlabel("Predicted Labels", fontsize=14, labelpad=10)
    plt.ylabel("True Labels", fontsize=14)
    plt.tick_params(left=False, bottom=False)
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    plt.show()

    return curves

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)