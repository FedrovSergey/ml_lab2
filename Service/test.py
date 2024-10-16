import numpy as np
import pandas as pd

def main():
    # Пример данных
    y_true = np.array([0, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 2, 2, 1])

    df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})

    # Точность (Accuracy)
    accuracy = (df['True'] == df['Predicted']).mean()
    print("Точность:", accuracy)


    # Чувствительность (Sensitivity)
    tp = df[((df['True'] == 1) & (df['Predicted'] == 1))].shape[0]
    fn = df[((df['True'] == 1) & (df['Predicted'] != 1))].shape[0]
    sensitivity = tp / (tp + fn)
    print("Чувствительность:", sensitivity)

    # Специфичность (Specificity)
    tn = df[((df['True'] != 1) & (df['Predicted'] != 1))].shape[0]
    fp = df[((df['True'] != 1) & (df['Predicted'] == 1))].shape[0]
    specificity = tn / (tn + fp)
    print("Специфичность:", specificity)

    # Балансировка ошибок (Balanced Error Rate)
    tp = df[((df['True'] == 1) & (df['Predicted'] == 1))].shape[0]
    fn = df[((df['True'] == 1) & (df['Predicted'] != 1))].shape[0]
    fp = df[((df['True'] != 1) & (df['Predicted'] == 1))].shape[0]
    tn = df[((df['True'] != 1) & (df['Predicted'] != 1))].shape[0]

    ber = (tp + fp) / (tp + fn + fp + tn)
    print("Балансировка ошибок:", ber)

    # Средняя точность (Mean Average Precision)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        map = 0
    else:
        map = (precision * recall) / (precision + recall)
    print("Средняя точность:", map)

    # Оценка F1 (F1 Score)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    print("Оценка F1:", f1)

    # Показатель Макдональда (McDonald's Delta)
    mdelta = ((precision + recall) / 2) - f1
    print("Показатель Макдональда:", mdelta)

    # MCC (Mathew Correlation Coefficient)
    tpos = df[((df['True'] == 1) & (df['Predicted'] == 1))].shape[0]
    fpos = df[((df['True'] != 1) & (df['Predicted'] == 1))].shape[0]
    tneg = df[((df['True'] != 1) & (df['Predicted'] != 1))].shape[0]
    fneg = df[((df['True'] == 1) & (df['Predicted'] != 1))].shape[0]

    mcc = (tpos * tneg - fpos * fneg) / np.sqrt((tpos + fpos) * (tpos + fneg) * (tneg + fpos) * (tneg + fneg))
    print("Коэффициент корреляции Маттеуса:", mcc)


if __name__ == "__main__":
    main()