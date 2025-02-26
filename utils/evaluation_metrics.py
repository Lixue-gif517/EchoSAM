
from hausdorff import hausdorff_distance
from sklearn.metrics import confusion_matrix

def dice_coeff(pred, target,device):
    smooth = 1.
    target = target.view(128,128)
    num = pred.size(0)
    m1 = pred.view(num, -1).to(device)  # Flatten
    m2 = target.view(num, -1).to(device)  # Flatten
    intersection = (m1 * m2).sum()
    return (2 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def compute_hausdorff_distance(pred_, target):
    # 计算Hausdorff距离，使用skimage库  numpy 格式
    hd = hausdorff_distance(pred_, target, distance="euclidean")
    return hd

# 计算真实值与预测值的IoU
def Iou(pred, true, device):
    true = true.view(128, 128)
    pred = pred.to(device)
    true = true.to(device)
    intersection = pred * true  # 计算交集  pred ∩ true
    temp = pred + true  # pred + true
    union = temp - intersection  # 计算并集：A ∪ B = A + B - A ∩ B
    smooth = 1e-8  # 防止分母为 0
    iou_score = intersection.sum() / (union.sum() + smooth)
    return iou_score


def F1_Score(pred, true, device, threshold=0.5):
    true = true.view(128, 128)
    pred = pred.cpu()
    true = true.cpu()
    y_pred_binary = pred
    # Flatten the arrays for confusion matrix computation
    y_true_flat = true.flatten()
    y_pred_flat = y_pred_binary.flatten()
    # Compute the confusion matrix
    # Compute the confusion matrix

    cm = confusion_matrix(y_true_flat, y_pred_flat)
    # Extract TP, FP, FN
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    # Compute Precision and Recall
    precision = TP / (TP + FP + 1e-7)  # Add small number to avoid division by zero
    recall = TP / (TP + FN + 1e-7)  # Add small number to avoid division by zero
    # Compute F1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1_score



if __name__ == '__main__':
    mask = 1
    pred = 1