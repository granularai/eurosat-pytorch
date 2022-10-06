import torch

def update_confusion_matrix(conf_matrix, outputs, labels):
    with torch.no_grad():
        _, pred = outputs.topk(1,1,True,True)
        
        pred = pred.t().tolist()[0]
        truth = labels.tolist()

        for tidx, pidx in zip(truth,pred):
            conf_matrix[tidx,pidx] += 1

def print_loss_metrics(losses, conf_matrix, classes):
    num_classes = len(classes)
    avg_loss = sum(losses)/len(losses)

    recall = [conf_matrix[i,i]/sum(conf_matrix[:,i]) for i in range(num_classes)]
    precision = [conf_matrix[i,i]/sum(conf_matrix[i,:]) for i in range(num_classes)]
    f1_score = [(2*precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(num_classes)]

    recall_map = {classes[idx]: recall[idx] for idx in range(num_classes)}
    precision_map = {classes[idx]: precision[idx] for idx in range(num_classes)}
    f1_score_map = {classes[idx]: f1_score[idx] for idx in range(num_classes)}

    metric_maps = {
        "precision": precision_map,
        "recall": recall_map,
        "f1 score": f1_score_map
    }

    avg_metric_map = {
        "precision": sum(precision)/num_classes,
        "recall": sum(recall)/num_classes,
        "f1 score": sum(f1_score)/num_classes,
        "accuracy": sum([conf_matrix[i,i] for i in range(num_classes)])/conf_matrix.sum()
    }

    print(f"\t\t loss: {avg_loss}")

    print("\t\t metrics:")
    
    print("\t\t\t confusion matrix:")
    print("\t\t\t "+f"{conf_matrix}".replace("\n","\n\t\t\t"))
    print("")

    print("\t\t\t class level metrics:")
    for met_key in metric_maps:
        print(f"\t\t\t\t {met_key}:")
        for cls, metric in metric_maps[met_key].items():
            print(f"\t\t\t\t\t {cls}: {metric}")
    print("")

    print("\t\t\t overall metrics:")
    for met_key in avg_metric_map:
        print(f"\t\t\t\t {met_key}: {avg_metric_map[met_key]}")
    print("")

    return avg_metric_map["f1 score"]